package verda

import _ "embed"

// DefaultPackerTemplate embeds the Verda Packer template used to create the
// golden OS volume that nnctl cloud deploy clones per instance.
//
//go:embed packer/ubuntu.pkr.hcl
var DefaultPackerTemplate string

type PackerTemplateFile struct {
	Path    string
	Content string
}

func PackerTemplateFiles() []PackerTemplateFile {
	return []PackerTemplateFile{
		{Path: "ubuntu.pkr.hcl", Content: DefaultPackerTemplate},
		{Path: "bootstrap.sh", Content: DefaultUserDataScript},
	}
}
