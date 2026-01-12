//
//  AIbyJCAppApp.swift
//  AIbyJCApp
//
//  Created by Jaanvi Chirimar on 1/11/26.
//

import SwiftUI
import SwiftData

@main
struct AIbyJCAppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: SavedPhoto.self)
    }
}
