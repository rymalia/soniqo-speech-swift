import SwiftUI

@main
struct DictateDemoApp: App {
    @StateObject private var viewModel = DictateViewModel()
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        MenuBarExtra {
            DictateMenuView(viewModel: viewModel)
                .onChange(of: viewModel.shouldShowHUD) {
                    if viewModel.shouldShowHUD {
                        openWindow(id: "dictate-hud")
                        viewModel.shouldShowHUD = false
                    }
                }
        } label: {
            Image(systemName: viewModel.isRecording ? "mic.fill" : "mic")
        }
        .menuBarExtraStyle(.window)

        Window("Dictate", id: "dictate-hud") {
            DictateHUDView(viewModel: viewModel)
                .frame(minWidth: 400, minHeight: 200)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 450, height: 300)
        .defaultPosition(.topTrailing)
    }
}
