// Command likeness is the face recognition server and its tooling.
package main

import (
	"fmt"
	"os"
	"runtime"

	"github.com/muhriddinsalohiddin/likeness"
)

const usage = `usage: likeness <command>

commands:
  version   print the version and exit
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	switch os.Args[1] {
	case "version", "-v", "--version":
		fmt.Printf("likeness %s (%s, %s/%s)\n", likeness.Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "likeness: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}
