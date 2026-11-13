//go:build !(linux || darwin)

package index

import "errors"

const canMap = false

func mappedBlock(dim, rows int) (*block, error) {
	return nil, errors.New("index: no anonymous mapping on this platform")
}
