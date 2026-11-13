package store

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync/atomic"
	"time"

	bolt "go.etcd.io/bbolt"
	bolterrors "go.etcd.io/bbolt/errors"
)

// Options say how to open a store file.
type Options struct {
	// Dim and Fingerprint describe the model whose vectors the store holds.
	// A new file records them; an existing file must agree with them, or
	// Open fails with ErrMismatch. Leave both zero to open an existing file
	// for whatever model it was made for.
	Dim         int
	Fingerprint string
	// Key, if set, is a KeySize-byte key: vectors and metadata are then
	// encrypted with AES-256-GCM. A store is encrypted or not from the day
	// it is created; the key cannot be added or dropped later. Subject ids
	// stay in the clear.
	Key []byte
	// ReadOnly opens the file without the right to change it. Several
	// processes can read a store at once, but not while one writes.
	ReadOnly bool
	// Timeout is how long to wait for another process to let go of the
	// file; zero means two seconds.
	Timeout time.Duration
}

const formatVersion = 1

var (
	bucketMeta      = []byte("meta")
	bucketSubjects  = []byte("subjects")
	bucketTemplates = []byte("templates")

	metaFormat      = []byte("format")
	metaDim         = []byte("dim")
	metaFingerprint = []byte("fingerprint")
	metaCreated     = []byte("created")
	metaCipher      = []byte("cipher")
	metaCheck       = []byte("check")

	checkBound = []byte("check")
	checkText  = []byte("likeness store")
)

// file is a Store in one bbolt database file.
type file struct {
	db       *bolt.DB
	info     Info
	seal     sealer
	readOnly bool
	closed   atomic.Bool

	subjects  atomic.Int64
	templates atomic.Int64
}

// Open opens the store file at path, creating it if it does not exist.
func Open(path string, o Options) (Store, error) {
	seal, err := newSealer(o.Key)
	if err != nil {
		return nil, err
	}
	timeout := o.Timeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	db, err := bolt.Open(path, 0o600, &bolt.Options{Timeout: timeout, ReadOnly: o.ReadOnly})
	if err != nil {
		if errors.Is(err, bolterrors.ErrTimeout) {
			return nil, fmt.Errorf("%w: %s", ErrBusy, path)
		}
		return nil, fmt.Errorf("store: %w", err)
	}
	s := &file{db: db, seal: seal, readOnly: o.ReadOnly}
	begin := db.Update
	if o.ReadOnly {
		begin = db.View
	}
	if err := begin(func(tx *bolt.Tx) error { return s.start(tx, o) }); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

// start reads the store's description, or writes it into a new file, and
// checks it against what the caller expects.
func (s *file) start(tx *bolt.Tx, o Options) error {
	meta := tx.Bucket(bucketMeta)
	if meta == nil {
		if o.ReadOnly {
			return errors.New("store: the file is not a store")
		}
		if o.Dim <= 0 || o.Fingerprint == "" {
			return errors.New("store: a new store needs the vector length and the model fingerprint")
		}
		return s.create(tx, o)
	}

	if v := meta.Get(metaFormat); len(v) != 4 || binary.LittleEndian.Uint32(v) != formatVersion {
		return fmt.Errorf("store: unsupported file format %v", v)
	}
	dim, created, cipher := meta.Get(metaDim), meta.Get(metaCreated), meta.Get(metaCipher)
	if len(dim) != 4 || len(created) != 8 || len(cipher) != 1 {
		return fmt.Errorf("%w: store description", ErrCorrupt)
	}
	s.info = Info{
		Dim:         int(binary.LittleEndian.Uint32(dim)),
		Fingerprint: string(meta.Get(metaFingerprint)),
		Created:     time.Unix(0, int64(binary.LittleEndian.Uint64(created))).UTC(),
		Encrypted:   cipher[0] == envelopeGCM,
	}

	switch {
	case s.info.Encrypted && !s.seal.encrypted():
		return fmt.Errorf("%w: the store is encrypted and no key was given", ErrKey)
	case !s.info.Encrypted && s.seal.encrypted():
		return fmt.Errorf("%w: a key was given, but the store was created without one", ErrKey)
	case s.info.Encrypted:
		if text, err := s.seal.open(checkBound, meta.Get(metaCheck)); err != nil || !bytes.Equal(text, checkText) {
			return fmt.Errorf("%w: wrong key", ErrKey)
		}
	}
	if (o.Dim != 0 && o.Dim != s.info.Dim) || (o.Fingerprint != "" && o.Fingerprint != s.info.Fingerprint) {
		return fmt.Errorf("%w: it holds %d-value vectors made by %q, not %d-value vectors made by %q",
			ErrMismatch, s.info.Dim, s.info.Fingerprint, o.Dim, o.Fingerprint)
	}

	subjects, templates := tx.Bucket(bucketSubjects), tx.Bucket(bucketTemplates)
	if subjects == nil || templates == nil {
		return fmt.Errorf("%w: store tables are missing", ErrCorrupt)
	}
	s.subjects.Store(int64(subjects.Stats().KeyN))
	s.templates.Store(int64(templates.Stats().KeyN))
	return nil
}

func (s *file) create(tx *bolt.Tx, o Options) error {
	meta, err := tx.CreateBucket(bucketMeta)
	if err != nil {
		return err
	}
	for _, name := range [][]byte{bucketSubjects, bucketTemplates} {
		if _, err := tx.CreateBucket(name); err != nil {
			return err
		}
	}
	now := time.Now().UTC()
	s.info = Info{Dim: o.Dim, Fingerprint: o.Fingerprint, Created: now, Encrypted: s.seal.encrypted()}
	cipher := byte(envelopePlain)
	if s.info.Encrypted {
		cipher = envelopeGCM
		check, err := s.seal.seal(checkBound, checkText)
		if err != nil {
			return err
		}
		if err := meta.Put(metaCheck, check); err != nil {
			return err
		}
	}
	for key, value := range map[string][]byte{
		string(metaFormat):      binary.LittleEndian.AppendUint32(nil, formatVersion),
		string(metaDim):         binary.LittleEndian.AppendUint32(nil, uint32(o.Dim)),
		string(metaFingerprint): []byte(o.Fingerprint),
		string(metaCreated):     binary.LittleEndian.AppendUint64(nil, uint64(now.UnixNano())),
		string(metaCipher):      {cipher},
	} {
		if err := meta.Put([]byte(key), value); err != nil {
			return err
		}
	}
	return nil
}

func (s *file) Info() Info { return s.info }

// ready is the common start of every operation.
func (s *file) ready(ctx context.Context, writing bool) error {
	if s.closed.Load() {
		return ErrClosed
	}
	if writing && s.readOnly {
		return ErrReadOnly
	}
	return ctx.Err()
}

func (s *file) Add(ctx context.Context, subject string, vector []float32, meta []byte) (Template, error) {
	added, err := s.AddMany(ctx, []Entry{{Subject: subject, Vector: vector, Meta: meta}})
	if err != nil {
		return Template{}, err
	}
	return added[0], nil
}

func (s *file) AddMany(ctx context.Context, entries []Entry) ([]Template, error) {
	if err := s.ready(ctx, true); err != nil {
		return nil, err
	}
	for i, e := range entries {
		if err := validEntry(e, s.info.Dim); err != nil {
			if len(entries) > 1 {
				return nil, fmt.Errorf("entry %d: %w", i, err)
			}
			return nil, err
		}
	}
	if len(entries) == 0 {
		return nil, nil
	}

	added := make([]Template, len(entries))
	var newSubjects int64
	err := s.db.Update(func(tx *bolt.Tx) error {
		newSubjects = 0
		subjects, templates := tx.Bucket(bucketSubjects), tx.Bucket(bucketTemplates)
		now := time.Now().UTC()
		for i, e := range entries {
			id, err := templates.NextSequence()
			if err != nil {
				return err
			}
			created, count := now.UnixNano(), uint32(0)
			if record := subjects.Get([]byte(e.Subject)); record != nil {
				if created, count, err = decodeSubject(record); err != nil {
					return err
				}
			} else {
				newSubjects++
			}
			if err := subjects.Put([]byte(e.Subject), encodeSubject(created, count+1)); err != nil {
				return err
			}
			key := templateKey(e.Subject, id)
			value, err := s.seal.seal(templateBound(key), encodeTemplate(now.UnixNano(), e.Meta, e.Vector))
			if err != nil {
				return err
			}
			if err := templates.Put(key, value); err != nil {
				return err
			}
			added[i] = Template{ID: id, Subject: e.Subject, Created: now, Vector: e.Vector, Meta: e.Meta}
		}
		return nil
	})
	if err != nil {
		return nil, s.failure(err)
	}
	s.subjects.Add(newSubjects)
	s.templates.Add(int64(len(entries)))
	return added, nil
}

func (s *file) Remove(ctx context.Context, subject string, template uint64) (bool, error) {
	if err := s.ready(ctx, true); err != nil {
		return false, err
	}
	if !nameable(subject) { // such a subject cannot exist
		return false, nil
	}
	found, emptied := false, false
	err := s.db.Update(func(tx *bolt.Tx) error {
		found, emptied = false, false
		subjects, templates := tx.Bucket(bucketSubjects), tx.Bucket(bucketTemplates)
		key := templateKey(subject, template)
		if templates.Get(key) == nil {
			return nil
		}
		found = true
		if err := templates.Delete(key); err != nil {
			return err
		}
		created, count, err := decodeSubject(subjects.Get([]byte(subject)))
		if err != nil {
			return err
		}
		if count <= 1 {
			emptied = true
			return subjects.Delete([]byte(subject))
		}
		return subjects.Put([]byte(subject), encodeSubject(created, count-1))
	})
	if err != nil {
		return false, s.failure(err)
	}
	if found {
		s.templates.Add(-1)
	}
	if emptied {
		s.subjects.Add(-1)
	}
	return found, nil
}

func (s *file) RemoveSubject(ctx context.Context, subject string) (int, error) {
	if err := s.ready(ctx, true); err != nil {
		return 0, err
	}
	if !nameable(subject) { // such a subject cannot exist
		return 0, nil
	}
	removed, existed := 0, false
	err := s.db.Update(func(tx *bolt.Tx) error {
		removed, existed = 0, false
		subjects, templates := tx.Bucket(bucketSubjects), tx.Bucket(bucketTemplates)
		if subjects.Get([]byte(subject)) == nil {
			return nil
		}
		existed = true
		prefix := templatePrefix(subject)
		var keys [][]byte
		c := templates.Cursor()
		for k, _ := c.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, _ = c.Next() {
			keys = append(keys, bytes.Clone(k))
		}
		for _, k := range keys {
			if err := templates.Delete(k); err != nil {
				return err
			}
		}
		removed = len(keys)
		return subjects.Delete([]byte(subject))
	})
	if err != nil {
		return 0, s.failure(err)
	}
	if existed {
		s.subjects.Add(-1)
		s.templates.Add(-int64(removed))
	}
	return removed, nil
}

func (s *file) Subject(ctx context.Context, id string) (Subject, error) {
	if err := s.ready(ctx, false); err != nil {
		return Subject{}, err
	}
	if !nameable(id) {
		return Subject{}, ErrNotFound
	}
	subject := Subject{ID: id}
	err := s.db.View(func(tx *bolt.Tx) error {
		record := tx.Bucket(bucketSubjects).Get([]byte(id))
		if record == nil {
			return ErrNotFound
		}
		created, _, err := decodeSubject(record)
		if err != nil {
			return err
		}
		subject.Created = time.Unix(0, created).UTC()
		prefix := templatePrefix(id)
		c := tx.Bucket(bucketTemplates).Cursor()
		for k, v := c.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = c.Next() {
			_, tid, err := splitTemplateKey(k)
			if err != nil {
				return err
			}
			plain, err := s.seal.open(templateBound(k), v)
			if err != nil {
				return err
			}
			at, meta, err := decodeTemplate(plain, s.info.Dim, nil)
			if err != nil {
				return err
			}
			subject.Templates = append(subject.Templates, TemplateInfo{ID: tid, Created: time.Unix(0, at).UTC(), Meta: bytes.Clone(meta)})
		}
		return nil
	})
	if err != nil {
		return Subject{}, s.failure(err)
	}
	return subject, nil
}

func (s *file) Subjects(ctx context.Context, after string, limit int) ([]Summary, error) {
	if err := s.ready(ctx, false); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 100
	}
	limit = min(limit, 1000)
	var list []Summary
	err := s.db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket(bucketSubjects).Cursor()
		k, v := c.First()
		if after != "" {
			if k, v = c.Seek([]byte(after)); k != nil && string(k) == after {
				k, v = c.Next()
			}
		}
		for ; k != nil && len(list) < limit; k, v = c.Next() {
			created, count, err := decodeSubject(v)
			if err != nil {
				return err
			}
			list = append(list, Summary{ID: string(k), Created: time.Unix(0, created).UTC(), Templates: int(count)})
		}
		return nil
	})
	if err != nil {
		return nil, s.failure(err)
	}
	return list, nil
}

func (s *file) Count(ctx context.Context) (subjects, templates int, err error) {
	if err := s.ready(ctx, false); err != nil {
		return 0, 0, err
	}
	return int(s.subjects.Load()), int(s.templates.Load()), nil
}

func (s *file) Load(ctx context.Context, fn func(Template) error) error {
	if err := s.ready(ctx, false); err != nil {
		return err
	}
	vector := make([]float32, s.info.Dim)
	err := s.db.View(func(tx *bolt.Tx) error {
		n := 0
		return tx.Bucket(bucketTemplates).ForEach(func(k, v []byte) error {
			if n++; n%256 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			subject, id, err := splitTemplateKey(k)
			if err != nil {
				return err
			}
			plain, err := s.seal.open(templateBound(k), v)
			if err != nil {
				return fmt.Errorf("%w (subject %q, template %d)", err, subject, id)
			}
			created, meta, err := decodeTemplate(plain, s.info.Dim, vector)
			if err != nil {
				return fmt.Errorf("%w (subject %q, template %d)", err, subject, id)
			}
			return fn(Template{ID: id, Subject: subject, Created: time.Unix(0, created).UTC(), Vector: vector, Meta: meta})
		})
	})
	return s.failure(err)
}

func (s *file) Backup(ctx context.Context, w io.Writer) error {
	if err := s.ready(ctx, false); err != nil {
		return err
	}
	return s.failure(s.db.View(func(tx *bolt.Tx) error {
		_, err := tx.WriteTo(w)
		return err
	}))
}

func (s *file) Close() error {
	if s.closed.Swap(true) {
		return nil
	}
	if err := s.db.Close(); err != nil {
		return fmt.Errorf("store: %w", err)
	}
	return nil
}

// failure translates the one bbolt error callers can meet by racing Close.
func (s *file) failure(err error) error {
	if errors.Is(err, bolterrors.ErrDatabaseNotOpen) {
		return ErrClosed
	}
	return err
}
