package client_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"

	"github.com/muhriddinsalohiddin/likeness/client"
	"github.com/muhriddinsalohiddin/likeness/index"
	"github.com/muhriddinsalohiddin/likeness/internal/enginetest"
	"github.com/muhriddinsalohiddin/likeness/internal/gallery"
	"github.com/muhriddinsalohiddin/likeness/internal/server"
	"github.com/muhriddinsalohiddin/likeness/store"
)

const (
	adminKey  = "an-admin-key-for-the-examples"
	searchKey = "a-search-key-for-the-examples"
)

func ExampleClient_Search() {
	// The examples talk to a server whose "pictures" are lines of text;
	// a real one is sent the bytes of a JPEG, PNG or WebP file.
	baseURL, stop := standIn("face:anna")
	defer stop()

	c, err := client.New(baseURL, client.WithKey(searchKey))
	if err != nil {
		fmt.Println(err)
		return
	}
	ctx := context.Background()

	for _, picture := range []string{"face:anna#2", "face:boris", "reject:no_face"} {
		res, err := c.Search(ctx, []byte(picture), client.SearchOptions{})
		switch {
		case client.IsOverloaded(err):
			// still busy after the client's own retries
			fmt.Println("the server is busy; ask again in a moment")
		case err != nil:
			fmt.Println("the search failed:", err)
		case res.Best() != nil:
			m := res.Best()
			fmt.Printf("this is %s, with a similarity of %.2f\n", m.Subject, m.Score)
		case res.Reject == client.ReasonNoFace:
			fmt.Println("there is no face in the picture")
		default:
			fmt.Println("nobody who is enrolled")
		}
	}
	// Output:
	// this is anna, with a similarity of 0.89
	// nobody who is enrolled
	// there is no face in the picture
}

func ExampleClient_Enrol() {
	baseURL, stop := standIn()
	defer stop()

	c, err := client.New(baseURL, client.WithKey(adminKey))
	if err != nil {
		fmt.Println(err)
		return
	}
	ctx := context.Background()

	enrol := func(subject, picture string) {
		enrolled, err := c.Enrol(ctx, subject, []byte(picture), client.EnrolOptions{})
		var refused *client.Error
		errors.As(err, &refused) // stays nil unless the server refused
		switch {
		case err == nil:
			fmt.Printf("%s now has %d template(s)\n", enrolled.Subject, enrolled.Templates)
		case client.IsRejected(err):
			// the picture will not do; Reject says what to ask the person for
			fmt.Printf("%s: take another picture (%s)\n", subject, refused.Reject)
		case client.IsDuplicate(err):
			fmt.Printf("%s: this face is already enrolled as %s\n", subject, refused.Subject)
		case refused != nil && refused.Code == client.CodeSubjectMismatch:
			fmt.Printf("%s: this is not the person enrolled under that id\n", subject)
		case client.IsOverloaded(err):
			fmt.Printf("%s: the server is busy; nothing was enrolled, try again later\n", subject)
		default:
			// The connection may have broken after the server enrolled the
			// face: look the subject up before sending the picture again.
			fmt.Printf("%s: %v\n", subject, err)
		}
	}
	enrol("employee-1042", "face:anna")
	enrol("employee-1042", "face:anna#2")
	enrol("employee-1042", "reject:blurry")
	enrol("employee-1042", "face:boris")
	enrol("employee-2000", "face:anna#3")
	// Output:
	// employee-1042 now has 1 template(s)
	// employee-1042 now has 2 template(s)
	// employee-1042: take another picture (blurry)
	// employee-1042: this is not the person enrolled under that id
	// employee-2000: this face is already enrolled as employee-1042
}

// standIn serves the API as likeness serve does, over an engine that reads
// its "pictures" as lines of text, so that the examples run without models.
// The pictures given are enrolled under the name they carry.
func standIn(enrolled ...string) (baseURL string, stop func()) {
	dir, err := os.MkdirTemp("", "likeness-example")
	if err != nil {
		log.Fatal(err)
	}
	engine := &enginetest.Fake{}
	info := engine.Info()
	st, err := store.Open(filepath.Join(dir, "faces.db"), store.Options{Dim: info.Dim, Fingerprint: info.Fingerprint})
	if err != nil {
		log.Fatal(err)
	}
	faces, err := gallery.Open(context.Background(), st, gallery.Options{Index: index.Options{OnHeap: true}})
	if err != nil {
		log.Fatal(err)
	}
	keys, err := server.NewKeys([]string{adminKey}, []string{searchKey}, nil)
	if err != nil {
		log.Fatal(err)
	}
	srv, err := server.New(server.Config{Engine: engine, Gallery: faces, Keys: keys})
	if err != nil {
		log.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())

	admin, err := client.New(ts.URL, client.WithKey(adminKey))
	if err != nil {
		log.Fatal(err)
	}
	for _, picture := range enrolled {
		if _, err := admin.Enrol(context.Background(), picture[len("face:"):], []byte(picture), client.EnrolOptions{}); err != nil {
			log.Fatal(err)
		}
	}
	return ts.URL, func() {
		ts.Close()
		faces.Close()
		os.RemoveAll(dir)
	}
}
