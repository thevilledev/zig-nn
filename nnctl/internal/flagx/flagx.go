package flagx

import (
	"flag"
	"fmt"
	"strings"
)

func SplitPassthrough(args []string) ([]string, []string) {
	for i, arg := range args {
		if arg == "--" {
			return args[:i], args[i+1:]
		}
	}
	return args, nil
}

func ParseInterspersed(fs *flag.FlagSet, args []string, valueFlags map[string]bool) error {
	reordered, err := ReorderInterspersed(args, valueFlags)
	if err != nil {
		return err
	}
	return fs.Parse(reordered)
}

func ReorderInterspersed(args []string, valueFlags map[string]bool) ([]string, error) {
	flags := make([]string, 0, len(args))
	positionals := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			positionals = append(positionals, arg)
			continue
		}

		name, hasInlineValue := ParseName(arg)
		wantsValue, known := valueFlags[name]
		if !known {
			flags = append(flags, arg)
			continue
		}

		flags = append(flags, arg)
		if wantsValue && !hasInlineValue {
			if i+1 >= len(args) {
				return nil, fmt.Errorf("flag needs an argument: -%s", name)
			}
			i++
			flags = append(flags, args[i])
		}
	}
	return append(flags, positionals...), nil
}

func ParseName(arg string) (string, bool) {
	arg = strings.TrimLeft(arg, "-")
	if arg == "" {
		return "", false
	}
	if index := strings.IndexByte(arg, '='); index >= 0 {
		return arg[:index], true
	}
	return arg, false
}

func HasAny(args []string, names ...string) bool {
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			continue
		}
		name, _ := ParseName(arg)
		for _, candidate := range names {
			if name == candidate {
				return true
			}
		}
	}
	return false
}
