// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "iOSEchoDemo",
    platforms: [.iOS("18.0"), .macOS("15.0")],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "iOSEchoDemo",
            dependencies: [
                .product(name: "KokoroTTS", package: "soniqo-speech-swift"),
                .product(name: "ParakeetASR", package: "soniqo-speech-swift"),
                .product(name: "SpeechVAD", package: "soniqo-speech-swift"),
                .product(name: "SpeechCore", package: "soniqo-speech-swift"),
                .product(name: "AudioCommon", package: "soniqo-speech-swift"),
            ],
            path: "iOSEchoDemo",
            exclude: ["Info.plist"]
        ),
    ]
)
