package vision

import "encoding/binary"

// exifOrientation returns the EXIF orientation tag (1..8) of a JPEG, or 1 when
// the file has none. Phones store portrait shots sideways and rely on this
// tag; ignoring it feeds rotated faces to the detector.
func exifOrientation(data []byte) int {
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1
	}
	for p := 2; p+4 <= len(data); {
		if data[p] != 0xFF {
			return 1
		}
		marker := data[p+1]
		switch {
		case marker == 0xFF: // fill byte
			p++
			continue
		case marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7): // no payload
			p += 2
			continue
		case marker == 0xDA || marker == 0xD9: // image data or end: no EXIF ahead
			return 1
		}
		size := int(binary.BigEndian.Uint16(data[p+2:]))
		if size < 2 || p+2+size > len(data) {
			return 1
		}
		if marker == 0xE1 {
			if o := tiffOrientation(data[p+4 : p+2+size]); o != 0 {
				return o
			}
		}
		p += 2 + size
	}
	return 1
}

// tiffOrientation reads tag 0x0112 from the first IFD of an "Exif\0\0"
// payload. It returns 0 when the payload is something else or malformed.
func tiffOrientation(app1 []byte) int {
	const header = "Exif\x00\x00"
	if len(app1) < len(header)+8 || string(app1[:len(header)]) != header {
		return 0
	}
	tiff := app1[len(header):]
	var order binary.ByteOrder
	switch string(tiff[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 0
	}
	if order.Uint16(tiff[2:]) != 42 {
		return 0
	}
	ifd := uint64(order.Uint32(tiff[4:]))
	if ifd+2 > uint64(len(tiff)) {
		return 0
	}
	entries := uint64(order.Uint16(tiff[ifd:]))
	for i := uint64(0); i < entries; i++ {
		e := ifd + 2 + 12*i
		if e+12 > uint64(len(tiff)) {
			return 0
		}
		if order.Uint16(tiff[e:]) != 0x0112 {
			continue
		}
		if order.Uint16(tiff[e+2:]) != 3 || order.Uint32(tiff[e+4:]) != 1 { // one SHORT
			return 0
		}
		if o := int(order.Uint16(tiff[e+8:])); o >= 1 && o <= 8 {
			return o
		}
		return 0
	}
	return 0
}
