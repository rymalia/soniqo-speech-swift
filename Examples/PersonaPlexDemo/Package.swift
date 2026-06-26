// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "PersonaPlexDemo",
    platforms: [.macOS("15.0")],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "PersonaPlexDemo",
            dependencies: [
                .product(name: "PersonaPlex", package: "soniqo-speech-swift"),
                .product(name: "Qwen3ASR", package: "soniqo-speech-swift"),
                .product(name: "AudioCommon", package: "soniqo-speech-swift"),
                .product(name: "SpeechVAD", package: "soniqo-speech-swift"),
            ],
            path: "PersonaPlexDemo",
            exclude: ["PersonaPlexDemo.entitlements", "Info.plist"]
        ),
        .testTarget(
            name: "PersonaPlexDemoTests",
            dependencies: ["PersonaPlexDemo"],
            path: "Tests"
        ),
    ]
)
