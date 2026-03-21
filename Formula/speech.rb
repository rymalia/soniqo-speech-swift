class Speech < Formula
  desc "AI speech models for Apple Silicon — ASR, TTS, speech-to-speech"
  homepage "https://github.com/soniqo/speech-swift"
  url "https://github.com/soniqo/speech-swift/releases/download/v0.0.7/audio-macos-arm64.tar.gz"
  sha256 "442d6857fe4b4f8c047d70a3bc5bffb1a06f1f441030702fb7c9dee39da15744"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on :macos

  def install
    libexec.install "audio", "mlx.metallib"
    bin.write_exec_script libexec/"audio"
  end

  test do
    assert_match "AI speech models", shell_output("#{bin}/audio --help")
  end
end
