package store

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
)

// Every stored value starts with one byte saying how to read the rest.
const (
	envelopePlain = 0 // the record follows
	envelopeGCM   = 1 // a 12-byte nonce and the AES-256-GCM sealed record follow
)

const recordVersion = 1

// sealer wraps and unwraps records. With no key it only adds the marker.
type sealer struct {
	aead cipher.AEAD
}

func newSealer(key []byte) (sealer, error) {
	if key == nil {
		return sealer{}, nil
	}
	if len(key) != KeySize {
		return sealer{}, fmt.Errorf("%w: must be %d bytes, got %d", ErrKey, KeySize, len(key))
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return sealer{}, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return sealer{}, err
	}
	return sealer{aead: aead}, nil
}

func (s sealer) encrypted() bool { return s.aead != nil }

// seal wraps a record. bound ties the result to the place it is stored at,
// so that a sealed record moved elsewhere no longer opens.
func (s sealer) seal(bound, record []byte) ([]byte, error) {
	if s.aead == nil {
		out := make([]byte, 0, 1+len(record))
		return append(append(out, envelopePlain), record...), nil
	}
	n := s.aead.NonceSize()
	out := make([]byte, 1+n, 1+n+len(record)+s.aead.Overhead())
	out[0] = envelopeGCM
	if _, err := rand.Read(out[1 : 1+n]); err != nil {
		return nil, err
	}
	return s.aead.Seal(out, out[1:1+n], record, bound), nil
}

func (s sealer) open(bound, stored []byte) ([]byte, error) {
	if len(stored) == 0 {
		return nil, fmt.Errorf("%w: empty value", ErrCorrupt)
	}
	switch stored[0] {
	case envelopePlain:
		if s.aead != nil {
			return nil, fmt.Errorf("%w: a plain record in an encrypted store", ErrCorrupt)
		}
		return stored[1:], nil
	case envelopeGCM:
		if s.aead == nil {
			return nil, fmt.Errorf("%w: an encrypted record in a plain store", ErrCorrupt)
		}
		n := s.aead.NonceSize()
		if len(stored) < 1+n+s.aead.Overhead() {
			return nil, fmt.Errorf("%w: sealed record cut short", ErrCorrupt)
		}
		record, err := s.aead.Open(nil, stored[1:1+n], stored[1+n:], bound)
		if err != nil {
			return nil, fmt.Errorf("%w: does not decrypt", ErrCorrupt)
		}
		return record, nil
	}
	return nil, fmt.Errorf("%w: unknown envelope %d", ErrCorrupt, stored[0])
}

// A template record is
//
//	version      1 byte
//	created      8 bytes, Unix nanoseconds
//	meta length  uvarint
//	meta
//	vector       4 bytes per value, IEEE 754
//
// all little-endian.
func encodeTemplate(created int64, meta []byte, vector []float32) []byte {
	out := make([]byte, 0, 1+8+binary.MaxVarintLen32+len(meta)+4*len(vector))
	out = append(out, recordVersion)
	out = binary.LittleEndian.AppendUint64(out, uint64(created))
	out = binary.AppendUvarint(out, uint64(len(meta)))
	out = append(out, meta...)
	for _, v := range vector {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
	}
	return out
}

// decodeTemplate reads a record. The vector is written into vector, which
// must have the store's dimension, unless it is nil; meta aliases record.
func decodeTemplate(record []byte, dim int, vector []float32) (created int64, meta []byte, err error) {
	if len(record) < 1+8+1 {
		return 0, nil, fmt.Errorf("%w: template record of %d bytes", ErrCorrupt, len(record))
	}
	if record[0] != recordVersion {
		return 0, nil, fmt.Errorf("%w: template record version %d", ErrCorrupt, record[0])
	}
	created = int64(binary.LittleEndian.Uint64(record[1:]))
	rest := record[9:]
	n, used := binary.Uvarint(rest)
	if used <= 0 || n > MaxMeta || uint64(len(rest)-used) < n {
		return 0, nil, fmt.Errorf("%w: bad metadata length", ErrCorrupt)
	}
	meta = rest[used : used+int(n)]
	raw := rest[used+int(n):]
	if len(raw) != 4*dim {
		return 0, nil, fmt.Errorf("%w: vector of %d bytes, want %d", ErrCorrupt, len(raw), 4*dim)
	}
	for i := range vector {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return created, meta, nil
}

// A subject record is its creation time and its number of templates.
func encodeSubject(created int64, templates uint32) []byte {
	out := make([]byte, 12)
	binary.LittleEndian.PutUint64(out, uint64(created))
	binary.LittleEndian.PutUint32(out[8:], templates)
	return out
}

func decodeSubject(record []byte) (created int64, templates uint32, err error) {
	if len(record) != 12 {
		return 0, 0, fmt.Errorf("%w: subject record of %d bytes", ErrCorrupt, len(record))
	}
	return int64(binary.LittleEndian.Uint64(record)), binary.LittleEndian.Uint32(record[8:]), nil
}

// Templates are keyed by subject, a zero byte, and the template id in
// big-endian, so that one subject's templates are adjacent and in id order.
// A subject id cannot contain a zero byte.
func templateKey(subject string, id uint64) []byte {
	key := make([]byte, 0, len(subject)+9)
	key = append(key, subject...)
	key = append(key, 0)
	return binary.BigEndian.AppendUint64(key, id)
}

func templatePrefix(subject string) []byte {
	return append([]byte(subject), 0)
}

func splitTemplateKey(key []byte) (subject string, id uint64, err error) {
	if len(key) < 10 || key[len(key)-9] != 0 {
		return "", 0, fmt.Errorf("%w: template key of %d bytes", ErrCorrupt, len(key))
	}
	return string(key[:len(key)-9]), binary.BigEndian.Uint64(key[len(key)-8:]), nil
}

// bound is the data a sealed template is tied to: its key.
func templateBound(key []byte) []byte {
	return append([]byte("template\x00"), key...)
}
