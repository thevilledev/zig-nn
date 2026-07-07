packer {
  required_plugins {
    verda = {
      version = ">= 0.1.0"
      source  = "github.com/thevilledev/verda"
    }
  }
}

variable "instance_type" {
  type    = string
  default = "V100"
}

variable "image" {
  type    = string
  default = "ubuntu-24.04"
}

variable "hostname" {
  type    = string
  default = "packer-verda-zig-nn"
}

source "verda-instance" "ubuntu" {
  instance_type = var.instance_type
  image         = var.image
  hostname      = var.hostname

  ssh_username = "root"

  keep_instance = true
}

build {
  sources = ["source.verda-instance.ubuntu"]

  provisioner "shell" {
    inline = [
      "cloud-init status --wait || true",
      "uname -a",
    ]
  }

  provisioner "shell" {
    script = "bootstrap.sh"
  }
}
