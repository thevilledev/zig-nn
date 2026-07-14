packer {
  required_plugins {
    verda = {
      version = ">= 0.1.1"
      source  = "github.com/thevilledev/verda"
    }
  }
}

variable "authorized_keys_file" {
  type        = string
  description = "Path to an authorized_keys-format file to bake into the OS volume for later boots."
}

variable "instance_type" {
  type    = string
  default = "CPU.4V.16G"
}

variable "image" {
  type    = string
  default = "ubuntu-24.04-cuda-13.0-open"
}

variable "hostname" {
  type    = string
  default = "packer-verda-zig-nn"
}

source "verda-instance" "ubuntu" {
  instance_type = var.instance_type
  image         = var.image
  hostname      = var.hostname

  ssh_username              = "root"
  ssh_clear_authorized_keys = true

  location_code = "FIN-01"
  contract      = "SPOT"

  artifact_type                  = "os_volume"
  artifact_volume_name           = "packer-verda-zig-nn-volume-root"
  artifact_volume_location_codes = ["FIN-02"]
}

build {
  sources = ["source.verda-instance.ubuntu"]

  provisioner "file" {
    source      = var.authorized_keys_file
    destination = "/tmp/verda_authorized_keys"
  }

  provisioner "shell" {
    inline = [
      "cloud-init status --wait || true",
      "uname -a",
      "install -d -m 0700 /root/.ssh",
      "touch /root/.ssh/authorized_keys",
      "chmod 0600 /root/.ssh/authorized_keys",
      "while IFS= read -r key; do [ -n \"$key\" ] || continue; grep -qxF \"$key\" /root/.ssh/authorized_keys || printf '%s\\n' \"$key\" >> /root/.ssh/authorized_keys; done < /tmp/verda_authorized_keys",
      "chown root:root /root/.ssh /root/.ssh/authorized_keys",
      "rm -f /tmp/verda_authorized_keys",
    ]
  }

  provisioner "shell" {
    script = "bootstrap.sh"
  }

  post-processor "manifest" {
    output = "packer-manifest.json"

    custom_data = {
      VolumeID            = build.VolumeID
      SourceOSVolumeID    = build.SourceOSVolumeID
      VolumeIDs           = jsonencode(build.VolumeIDs)
      VolumeIDsByLocation = jsonencode(build.VolumeIDsByLocation)
    }
  }
}
