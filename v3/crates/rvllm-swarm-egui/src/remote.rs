use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::ipc::{now_ms, Cmd, LogLevel, LogLine};
use crate::state::AppState;

const REMOTE_POLL_MS: u64 = 180;
const REMOTE_RECONNECT_MS: u64 = 800;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RemoteRequest {
    Snapshot,
    Command { cmd: Cmd },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RemoteResponse {
    Snapshot { state: Box<AppState> },
    Ack,
    Error { message: String },
}

pub fn serve_forever(
    addr: &str,
    state: Arc<RwLock<AppState>>,
    cmd_tx: Sender<Cmd>,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr)?;
    tracing::info!("swarmd listening on {}", listener.local_addr()?);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = Arc::clone(&state);
                let cmd_tx = cmd_tx.clone();
                std::thread::Builder::new()
                    .name("swarm-remote-client".into())
                    .spawn(move || {
                        if let Err(err) = handle_client(stream, state, cmd_tx) {
                            tracing::warn!("remote client closed: {err}");
                        }
                    })?;
            }
            Err(err) => {
                tracing::warn!("remote accept failed: {err}");
            }
        }
    }
    Ok(())
}

pub fn spawn_client(state: Arc<RwLock<AppState>>, cmd_rx: Receiver<Cmd>, addr: String) {
    std::thread::Builder::new()
        .name("swarm-remote-poll".into())
        .spawn(move || remote_client_loop(state, cmd_rx, addr))
        .expect("failed to spawn remote client");
}

fn remote_client_loop(state: Arc<RwLock<AppState>>, cmd_rx: Receiver<Cmd>, addr: String) {
    let mut was_connected = false;
    loop {
        match TcpStream::connect(&addr) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                if !was_connected {
                    push_log(
                        &state,
                        LogLevel::Info,
                        "remote",
                        format!("connected to remote swarm at {addr}"),
                    );
                    was_connected = true;
                }
                if let Err(err) = run_remote_session(&state, &cmd_rx, stream) {
                    push_log(
                        &state,
                        LogLevel::Warn,
                        "remote",
                        format!("remote session dropped: {err}"),
                    );
                }
            }
            Err(err) => {
                if was_connected {
                    push_log(
                        &state,
                        LogLevel::Warn,
                        "remote",
                        format!("remote connect failed: {err}"),
                    );
                    was_connected = false;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(REMOTE_RECONNECT_MS));
    }
}

fn run_remote_session(
    state: &Arc<RwLock<AppState>>,
    cmd_rx: &Receiver<Cmd>,
    mut writer: TcpStream,
) -> anyhow::Result<()> {
    let reader_stream = writer.try_clone()?;
    let mut reader = BufReader::new(reader_stream);
    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            write_request(&mut writer, &RemoteRequest::Command { cmd })?;
            match read_response(&mut reader)? {
                RemoteResponse::Ack => {}
                RemoteResponse::Error { message } => anyhow::bail!(message),
                RemoteResponse::Snapshot { .. } => anyhow::bail!("unexpected snapshot ack"),
            }
        }
        write_request(&mut writer, &RemoteRequest::Snapshot)?;
        match read_response(&mut reader)? {
            RemoteResponse::Snapshot { state: snapshot } => {
                *state.write() = *snapshot;
            }
            RemoteResponse::Ack => anyhow::bail!("unexpected ack to snapshot"),
            RemoteResponse::Error { message } => anyhow::bail!(message),
        }
        std::thread::sleep(Duration::from_millis(REMOTE_POLL_MS));
    }
}

fn handle_client(
    stream: TcpStream,
    state: Arc<RwLock<AppState>>,
    cmd_tx: Sender<Cmd>,
) -> anyhow::Result<()> {
    let _ = stream.set_nodelay(true);
    let reader_stream = stream.try_clone()?;
    let mut reader = BufReader::new(reader_stream);
    let mut writer = stream;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let request: RemoteRequest = serde_json::from_str(line.trim_end())?;
        let response = match request {
            RemoteRequest::Snapshot => RemoteResponse::Snapshot {
                state: Box::new(state.read().clone()),
            },
            RemoteRequest::Command { cmd } => match cmd_tx.send(cmd) {
                Ok(()) => RemoteResponse::Ack,
                Err(err) => RemoteResponse::Error {
                    message: err.to_string(),
                },
            },
        };
        serde_json::to_writer(&mut writer, &response)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
}

fn write_request(writer: &mut TcpStream, request: &RemoteRequest) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *writer, request)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn read_response(reader: &mut BufReader<TcpStream>) -> anyhow::Result<RemoteResponse> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        anyhow::bail!("remote closed connection");
    }
    Ok(serde_json::from_str(line.trim_end())?)
}

fn push_log(state: &Arc<RwLock<AppState>>, level: LogLevel, source: &str, message: String) {
    let mut guard = state.write();
    guard.log.push(LogLine {
        ts_ms: now_ms(),
        level,
        source: source.into(),
        message,
    });
}
