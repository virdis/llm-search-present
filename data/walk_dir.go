package data

import (
	"os"
	"path/filepath"
)

// Function to list all files in a given directory
func listFilesInDirectory(dirPath string) ([]string, error) {
	var fileNames []string

	err := filepath.WalkDir(dirPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() { // Only include files in the list
			fileNames = append(fileNames, path)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return fileNames, nil
}
