import SwiftUI

struct DictateMenuView: View {
    @ObservedObject var viewModel: DictateViewModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if viewModel.isLoading {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(viewModel.loadingStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
            } else if !viewModel.modelLoaded {
                Button("Load Models") {
                    Task { await viewModel.loadModels() }
                }
                .padding(.horizontal)
            } else {
                Button {
                    if !viewModel.isRecording {
                        openWindow(id: "dictate-hud")
                    }
                    viewModel.toggleRecording()
                } label: {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .foregroundStyle(viewModel.isRecording ? .red : .accentColor)
                        Text(viewModel.isRecording ? "Stop" : "Start Dictation")
                    }
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
                .padding(.horizontal)

                if viewModel.isRecording {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(viewModel.isSpeechActive ? .green : .orange)
                            .frame(width: 6, height: 6)
                        Text(viewModel.isSpeechActive ? "Speech" : "Silence")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(viewModel.wordCount)w")
                            .font(.caption2).monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal)
                }

                if !viewModel.fullText.isEmpty {
                    Divider()

                    ScrollView {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(Array(viewModel.sentences.enumerated()), id: \.offset) { _, sentence in
                                Text(sentence)
                                    .font(.system(.body, design: .rounded))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            if !viewModel.partialText.isEmpty {
                                Text(viewModel.partialText)
                                    .font(.system(.body, design: .rounded))
                                    .foregroundStyle(.secondary)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .padding(.horizontal)
                    }
                    .frame(maxWidth: 320, maxHeight: 150)

                    HStack(spacing: 8) {
                        Button("Copy") { viewModel.copyToClipboard() }
                        Button("Paste") { viewModel.pasteToFrontApp() }
                        Spacer()
                        Button("Clear") { viewModel.clearText() }
                    }
                    .padding(.horizontal)
                }
            }

            if let error = viewModel.errorMessage {
                Text(error).font(.caption).foregroundStyle(.red).padding(.horizontal)
            }

            Divider()
            Button("Quit") { NSApplication.shared.terminate(nil) }
                .keyboardShortcut("q").padding(.horizontal)
        }
        .padding(.vertical, 8)
        .frame(minWidth: 250)
        // Models loaded from DictateDemoApp.swift .task modifier
    }
}
