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

variable "base_url" {
  type        = string
  default     = ""
  description = "Optional Verda API base URL."
}

variable "build_location_code" {
  type    = string
  default = "FIN-01"
}

variable "build_market" {
  type    = string
  default = "spot"

  validation {
    condition     = contains(["spot", "on-demand"], var.build_market)
    error_message = "Build market must be spot or on-demand."
  }
}

variable "artifact_volume_name" {
  type    = string
  default = "packer-verda-zig-nn-volume-root"
}

variable "artifact_volume_location_codes" {
  type    = list(string)
  default = ["FIN-02"]
}

source "verda-instance" "ubuntu" {
  instance_type = var.instance_type
  image         = var.image
  hostname      = var.hostname

  base_url = var.base_url

  ssh_username              = "root"
  ssh_clear_authorized_keys = true

  location_code = var.build_location_code
  contract      = var.build_market == "spot" ? "SPOT" : "PAY_AS_YOU_GO"
  is_spot       = var.build_market == "spot"

  artifact_type                  = "os_volume"
  artifact_volume_name           = var.artifact_volume_name
  artifact_volume_location_codes = var.artifact_volume_location_codes
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
