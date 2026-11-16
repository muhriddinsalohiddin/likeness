package server

import (
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"strings"
)

// Scope is what a key may do.
type Scope uint8

const (
	// ScopeMetrics reads /metrics and nothing else: the key to give a
	// monitoring system.
	ScopeMetrics Scope = 1 << iota
	// ScopeSearch recognises faces: search, verify and info.
	ScopeSearch
	// ScopeAdmin enrols and removes people.
	ScopeAdmin
)

// MinKeyLength is the shortest API key the server accepts, in bytes.
const MinKeyLength = 16

const maxKeyLength = 512

// Keys is a set of API keys. Only their SHA-256 sums are kept.
type Keys struct {
	known []key
}

type key struct {
	sum    [sha256.Size]byte
	grants Scope
	// label is what the logs call the key, "admin-1": enough to tell which
	// one was used without giving anything away about it.
	label string
}

// NewKeys builds a key set. An admin key may do everything, a search key
// may only recognise, a metrics key may only read the metrics. A key must
// be at least MinKeyLength visible ASCII characters and appear once.
func NewKeys(admin, search, metrics []string) (Keys, error) {
	var k Keys
	kinds := []struct {
		name   string
		grants Scope
		keys   []string
	}{
		{"admin", ScopeAdmin | ScopeSearch | ScopeMetrics, admin},
		{"search", ScopeSearch, search},
		{"metrics", ScopeMetrics, metrics},
	}
	for _, kind := range kinds {
		for i, secret := range kind.keys {
			label := fmt.Sprintf("%s-%d", kind.name, i+1)
			if err := validKey(secret); err != nil {
				return Keys{}, fmt.Errorf("server: key %s %w", label, err)
			}
			sum := sha256.Sum256([]byte(secret))
			for _, other := range k.known {
				if other.sum == sum {
					return Keys{}, fmt.Errorf("server: key %s is the same as %s", label, other.label)
				}
			}
			k.known = append(k.known, key{sum: sum, grants: kind.grants, label: label})
		}
	}
	return k, nil
}

func validKey(secret string) error {
	switch {
	case len(secret) < MinKeyLength:
		return fmt.Errorf("is shorter than %d characters", MinKeyLength)
	case len(secret) > maxKeyLength:
		return fmt.Errorf("is longer than %d characters", maxKeyLength)
	}
	for i := 0; i < len(secret); i++ {
		if c := secret[i]; c <= ' ' || c > '~' {
			return errors.New("contains a space or a character outside visible ASCII")
		}
	}
	return nil
}

// Len is the number of keys in the set.
func (k Keys) Len() int { return len(k.known) }

// find looks a presented token up. Every key is compared, and compared in
// constant time, so the answer takes equally long whichever key matches or
// none does.
func (k Keys) find(token string) (grants Scope, label string, ok bool) {
	sum := sha256.Sum256([]byte(token))
	for i := range k.known {
		if subtle.ConstantTimeCompare(sum[:], k.known[i].sum[:]) == 1 {
			grants, label, ok = k.known[i].grants, k.known[i].label, true
		}
	}
	return grants, label, ok
}

// bearer extracts the token from an Authorization header.
func bearer(header string) (string, bool) {
	const scheme = "bearer "
	if len(header) <= len(scheme) || !strings.EqualFold(header[:len(scheme)], scheme) {
		return "", false
	}
	token := strings.TrimSpace(header[len(scheme):])
	return token, token != ""
}
