// MLX Master Trainer — Tauri v2 menu-bar shell.
// Generalized from the BigBugAI Bro studio shell (same proven pattern). Deliberately thin:
//   (1) live in the menu bar (Accessory policy, tray toggle),
//   (2) spawn the FastAPI backend (backend/server.py) as a child and kill it on quit,
//   (3) show a window that loads the bundled frontend, then navigate it to the backend origin
//       (127.0.0.1:8808) so the panels' fetches run same-origin.
// NO engine code here — the backend is the shell over core/; this is just the native wrapper.

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{
    image::Image,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, RunEvent,
};

const TRAY_ICON: &[u8] = include_bytes!("../icons/tray-idle-Template.png");
const PORT: &str = "127.0.0.1:8808";

/// Holds the backend child so we can reap it on exit.
struct Backend(Mutex<Option<Child>>);

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/".into()))
}

/// venv python + server path. Overridable via env for non-default checkouts.
fn backend_paths() -> (PathBuf, PathBuf) {
    let root = std::env::var("MMT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home().join("mlx-master-trainer"));
    let py = std::env::var("MMT_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join(".venv/bin/python"));
    let server = root.join("backend/server.py");
    (py, server)
}

fn port_open() -> bool {
    TcpStream::connect_timeout(&PORT.parse().unwrap(), Duration::from_millis(300)).is_ok()
}

/// Spawn the backend unless something is already serving the port (e.g. a dev run).
fn spawn_backend(app: &AppHandle) -> Option<Child> {
    if port_open() {
        return None; // already up; don't double-bind
    }
    // 1) shipped app: the PyInstaller-frozen sidecar bundled under Resources — NO user Python needed.
    //    (no args -> the FastAPI server; the frozen binary defaults its data root to ~/.mlx-master-trainer)
    if let Ok(res) = app.path().resource_dir() {
        let sidecar = res.join("mmt-backend").join("mmt-backend");
        if sidecar.exists() {
            return Command::new(sidecar).spawn().ok();
        }
    }
    // 2) dev fallback: the venv python + server.py (MMT_ROOT / MMT_PYTHON overridable)
    let (py, server) = backend_paths();
    if !py.exists() || !server.exists() {
        eprintln!("backend not found: no bundled sidecar AND {} / {}", py.display(), server.display());
        return None;
    }
    Command::new(py).arg(server).spawn().ok()
}

fn toggle_window(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let visible = win.is_visible().unwrap_or(false);
        if visible {
            let _ = win.hide();
        } else {
            let _ = win.show();
            let _ = win.set_focus();
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_positioner::init())
        .manage(Backend(Mutex::new(None)))
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let child = spawn_backend(app.handle());
            *app.state::<Backend>().0.lock().unwrap() = child;

            // The bundled page loads at the tauri:// origin, from which WKWebView blocks fetch()
            // to http://127.0.0.1:8808. Once the backend is serving, navigate the window to the
            // backend origin so the whole app runs same-origin and the panels' fetches work.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                for _ in 0..120 {
                    if port_open() {
                        let h = handle.clone();
                        let _ = handle.run_on_main_thread(move || {
                            if let Some(w) = h.get_webview_window("main") {
                                if let Ok(url) = "http://127.0.0.1:8808/".parse() {
                                    let _ = w.navigate(url);
                                }
                                let _ = w.center();
                            }
                        });
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(500));
                }
            });

            let open = MenuItem::with_id(app, "open", "Open Trainer", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let menu = Menu::with_items(app, &[&open, &sep, &quit])?;

            let mut builder = TrayIconBuilder::with_id("main")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => toggle_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    tauri_plugin_positioner::on_tray_event(tray.app_handle(), &event);
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        toggle_window(tray.app_handle());
                    }
                });
            if let Ok(img) = Image::from_bytes(TRAY_ICON) {
                builder = builder.icon(img).icon_as_template(true);
            }
            builder.build(app)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building mlx-master-trainer")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } | RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Backend>().0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
