# MODUL 4.2 — MCP & A2A: Infrastruktur Standar untuk Agent

> **Estimasi waktu**: 4–5 jam
> **Prerequisite**: Modul 4.1

---

### 🎯 Tujuan Belajar

- Memahami masalah integrasi N×M yang dipecahkan oleh MCP
- Menjelaskan arsitektur client-server MCP dan tiga primitif-nya (Tools, Resources, Prompts)
- Memahami perbedaan dan hubungan komplementer antara MCP dan A2A
- Mampu mengkonfigurasi MCP server dan menghubungkannya ke agent

---

### 🤔 Kenapa Ini Penting?

Di Modul 4.1, kamu sudah belajar bahwa agent perlu **tools** untuk berinteraksi dengan dunia luar. Tapi di sana, tools didefinisikan langsung dalam kode Python — satu agent, satu set tools, semuanya hardcoded.

Di dunia nyata, masalahnya jauh lebih kompleks:
- Kamu punya **puluhan tools** (database, API, file system, web search, email...)
- Kamu punya **beberapa agent** yang semuanya butuh akses ke tools berbeda
- Tools dikembangkan oleh tim yang berbeda, mungkin di bahasa pemrograman yang berbeda
- Setiap model (Claude, Gemini, GPT, Qwen) punya format function calling yang **sedikit berbeda**

Tanpa standar, kamu butuh integrasi kustom untuk setiap kombinasi agent × tool. Kalau punya 5 agent dan 10 tools, itu **50 integrasi kustom**. Ini masalah N×M yang tidak skalabel.

**MCP** mengubah ini menjadi masalah **N + M**: setiap tool cukup mengimplementasikan satu standar, dan setiap agent cukup mendukung satu standar. Selesai.

---

### 📖 Model Context Protocol (MCP): "USB-C untuk AI"

MCP adalah protokol komunikasi terbuka yang menstandardisasi cara AI agent terhubung ke data dan tools eksternal. Awalnya diciptakan oleh Anthropic pada 2024, sekarang dikelola oleh **Agentic AI Foundation (AAIF)** di bawah Linux Foundation — artinya ini bukan milik satu perusahaan, tapi standar industri terbuka.

Analogi paling tepat: **USB-C**. Sebelum USB-C, setiap perangkat punya port charger sendiri. Setelah USB-C, satu kabel untuk semuanya. MCP melakukan hal yang sama untuk koneksi agent-to-tool.

#### Arsitektur: Client-Server

```
┌──────────────────────────────────────┐
│  MCP HOST (aplikasi AI)              │
│  Contoh: Claude Desktop, Cursor,     │
│          VS Code, custom agent       │
│                                      │
│  ┌────────────┐  ┌────────────┐     │
│  │ MCP Client │  │ MCP Client │     │  ← Satu client per server
│  └─────┬──────┘  └─────┬──────┘     │
└────────┼───────────────┼─────────────┘
         │               │
    JSON-RPC 2.0    JSON-RPC 2.0
         │               │
┌────────▼────────┐ ┌────▼────────────┐
│  MCP SERVER A   │ │  MCP SERVER B   │
│  (contoh:       │ │  (contoh:       │
│   PostgreSQL)   │ │   GitHub API)   │
│                 │ │                 │
│  Ekspor:        │ │  Ekspor:        │
│  - Tools        │ │  - Tools        │
│  - Resources    │ │  - Resources    │
│  - Prompts      │ │  - Prompts      │
└─────────────────┘ └─────────────────┘
```

Tiga komponen utama:

- **MCP Host**: Aplikasi AI yang digunakan user — Claude Desktop, Cursor, VS Code, atau agent custom buatanmu
- **MCP Client**: Komponen di dalam host yang mengelola koneksi ke server-server MCP
- **MCP Server**: Service ringan yang mengekspos data atau fungsionalitas spesifik ke host. Komunikasi menggunakan **JSON-RPC 2.0** — standar web yang sudah mapan

---

### 📖 Tiga Primitif MCP

Setiap MCP server mengekspos kemampuannya melalui tiga primitif standar:

#### 1. Tools — "Apa yang bisa saya *lakukan*?"

Tools adalah fungsi yang bisa dipanggil oleh agent untuk melakukan aksi. Ini yang paling mirip dengan function calling di Modul 4.1, tapi sekarang dalam format standar.

Contoh tools yang diekspos oleh MCP server PostgreSQL:
- `query(sql: string)` — jalankan SQL query
- `list_tables()` — tampilkan semua tabel
- `describe_table(name: string)` — tampilkan skema tabel

#### 2. Resources — "Data apa yang bisa saya *baca*?"

Resources adalah data pasif yang bisa dibaca oleh agent — file, log, database row, respons API. Bedanya dengan tools: resources tidak mengubah state, hanya membaca.

Contoh resources dari MCP server file system:
- `file:///path/to/document.pdf` — konten dokumen
- `file:///logs/app.log` — log aplikasi

#### 3. Prompts — "Template apa yang tersedia?"

Prompts adalah template yang bisa dipakai untuk memandu perilaku agent di workflow tertentu. Ini seperti "resep" yang sudah disiapkan oleh server.

Contoh: MCP server untuk code review bisa menyediakan prompt template `review_pull_request` yang sudah berisi instruksi lengkap tentang apa yang harus dicek.

---

### 📖 MCP dalam Praktik: Konfigurasi

Konfigurasi MCP server dilakukan melalui JSON config. Setiap entry mendefinisikan satu server:

```json
{
  "mcpServers": {
    "postgres": {
      "command": "uvx",
      "args": ["mcp-server-postgres", "--connection-string", "postgresql://localhost/mydb"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx..."
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

Perhatikan:
- Setiap server adalah proses terpisah yang dijalankan oleh host
- Komunikasi via **stdio** (stdin/stdout) — simple dan universal
- Server bisa ditulis dalam bahasa apa pun (Python, TypeScript, Go, Rust...)
- SDK resmi tersedia untuk **Python** dan **TypeScript**

#### Ekosistem MCP Server yang Sudah Ada

Karena MCP sudah menjadi standar de facto, ekosistem server-nya sangat luas:

| Kategori | Contoh Server |
|---|---|
| **Database** | PostgreSQL, Supabase, MongoDB |
| **Developer Tools** | GitHub, GitLab, Jira |
| **Produktivitas** | Slack, Google Drive, Notion |
| **Desain** | Figma, Webflow |
| **File System** | Local filesystem, S3 |
| **Web** | Brave Search, Fetch URL |

Kamu tidak perlu menulis integrasi sendiri untuk tools umum — cukup pasang MCP server yang sudah ada.

---

### 📖 Agent2Agent Protocol (A2A): Ketika Agent Bicara ke Agent

MCP menyelesaikan masalah **agent-to-tool**. Tapi ada masalah lain: bagaimana jika kamu punya **beberapa agent** yang perlu berkolaborasi, dan mereka dibangun di framework atau platform yang berbeda?

Misalnya:
- **Agent Inventory** (dibangun di LangGraph, jalan di Google Cloud)
- **Agent Shipping** (dibangun di CrewAI, jalan di AWS)
- **Agent Payment** (dibangun di custom Python, jalan di on-premise)

Bagaimana mereka berkomunikasi? Tanpa standar, kamu butuh "glue code" kustom untuk setiap pasangan agent.

**A2A (Agent2Agent)** — diperkenalkan oleh Google pada April 2025 dan sekarang di bawah Linux Foundation — menyelesaikan ini.

#### Arsitektur A2A

```
┌──────────────┐                      ┌──────────────┐
│  Agent A     │  ──── A2A ────────▶  │  Agent B     │
│  (Client)    │                      │  (Remote)    │
│              │  1. Discover via     │              │
│  "Saya butuh │     Agent Card      │  "Saya bisa  │
│   cek stok"  │  2. Create Task     │   cek stok!" │
│              │  3. Receive         │              │
│              │     Artifact        │              │
└──────────────┘                      └──────────────┘
```

Tiga konsep utama A2A:

**1. Agent Cards** — "Résumé" agent
Setiap agent mempublikasikan Agent Card dalam format JSON yang mendeskripsikan:
- Apa kemampuannya
- Aksi apa yang tersedia
- Bagaimana cara mengaksesnya

Agent Card adalah mekanisme *discovery* — bagaimana satu agent "menemukan" agent lain yang bisa membantunya.

**2. Tasks** — Unit kerja
Ketika Agent A butuh bantuan Agent B, ia membuat *task* dengan lifecycle yang jelas: `created → running → paused → completed / failed`. Ini memungkinkan operasi yang berjalan lama — agent tidak harus menunggu jawaban dalam hitungan detik.

**3. Artifacts** — Hasil kerja
Setelah task selesai, agent mengembalikan *artifacts* — output atau hasil yang bisa digunakan oleh agent pemanggil.

#### MCP vs A2A: Komplementer, Bukan Kompetitor

Ini pertanyaan yang sering membingungkan. Jawaban singkatnya:

```
┌─────────────────────────────────┐
│           AGENT                 │
│                                 │
│  Otak (LLM) ──── MCP ────→ Tools, Data, APIs
│       │                         │
│       │                         │
│       └──────── A2A ────→ Agent lain (yang punya tools sendiri)
│                                 │
└─────────────────────────────────┘
```

| | MCP | A2A |
|---|---|---|
| **Menghubungkan** | Agent ↔ Tool/Data | Agent ↔ Agent |
| **Analogi** | "Bagaimana saya pakai database ini?" | "Saya butuh Agent Shipping untuk booking pengiriman" |
| **Tujuan** | Standardisasi akses ke *resource* | Standardisasi kolaborasi antar *agent* |
| **Visibilitas** | Agent melihat detail internal tools | Agent *tidak perlu* tahu detail internal agent lain (opacity) |

Keduanya dibangun di atas teknologi web standar (HTTP, JSON-RPC, SSE) dan keduanya di bawah Linux Foundation.

---

### 💻 Kode: Membangun MCP Server Sederhana (Python)

```python
# mcp_cuaca_server.py
# MCP server sederhana yang menyediakan tool "cek cuaca"
# Jalankan: uv run mcp_cuaca_server.py

from mcp.server.fastmcp import FastMCP

# Inisialisasi MCP server
mcp = FastMCP("Cuaca Indonesia")

# --- Definisikan TOOL ---
@mcp.tool()
def cek_cuaca(kota: str) -> str:
    """Cek kondisi cuaca terkini di kota Indonesia tertentu.
    
    Args:
        kota: Nama kota di Indonesia (contoh: "Jakarta", "Bandung", "Surabaya")
    """
    # Dalam produksi, ini akan memanggil API cuaca seperti OpenWeatherMap
    # Untuk demo, kita gunakan data hardcoded
    data_cuaca = {
        "jakarta": "☀️ Cerah, 32°C, kelembaban 65%",
        "bandung": "🌤️ Berawan sebagian, 24°C, kelembaban 72%",
        "surabaya": "🌧️ Hujan ringan, 29°C, kelembaban 80%",
        "yogyakarta": "☀️ Cerah, 30°C, kelembaban 60%",
        "bali": "🌤️ Berawan, 28°C, kelembaban 75%",
    }
    
    kota_lower = kota.lower()
    if kota_lower in data_cuaca:
        return f"Cuaca di {kota}: {data_cuaca[kota_lower]}"
    else:
        return f"Data cuaca untuk '{kota}' tidak tersedia. Kota yang tersedia: {', '.join(k.title() for k in data_cuaca)}"

# --- Definisikan RESOURCE ---
@mcp.resource("cuaca://daftar-kota")
def daftar_kota() -> str:
    """Daftar kota yang tersedia untuk pengecekan cuaca."""
    return "Kota tersedia: Jakarta, Bandung, Surabaya, Yogyakarta, Bali"

# --- Definisikan PROMPT ---
@mcp.prompt()
def laporan_cuaca(kota: str) -> str:
    """Template prompt untuk membuat laporan cuaca lengkap."""
    return f"""Buatkan laporan cuaca untuk kota {kota} dengan format berikut:
    
    1. Kondisi saat ini (gunakan tool cek_cuaca)
    2. Rekomendasi pakaian
    3. Rekomendasi aktivitas outdoor
    
    Gunakan bahasa Indonesia yang natural dan ramah."""

if __name__ == "__main__":
    mcp.run()
```

Untuk menghubungkan server ini ke Claude Desktop, tambahkan ke konfigurasi:

```json
{
  "mcpServers": {
    "cuaca": {
      "command": "uv",
      "args": ["run", "mcp_cuaca_server.py"]
    }
  }
}
```

Sekarang Claude bisa langsung menggunakan tool `cek_cuaca`, membaca resource `daftar_kota`, dan menggunakan prompt template `laporan_cuaca` — tanpa kode integrasi tambahan.

> **Yang perlu kamu perhatikan**: Bandingkan kode di atas dengan kode function calling di Modul 4.1. Di Modul 4.1, kamu harus menulis loop ReAct, mengirim tool definitions ke API, mengeksekusi tool, dan mengembalikan hasilnya secara manual. Dengan MCP, semua itu ditangani oleh protokol. Kamu hanya perlu menulis logika tool-nya.

---

### 📖 Deep-Dive: MCP Transports & Keamanan di Produksi

Memindahkan MCP dari eksperimen lokal (Claude Desktop) ke sistem produksi enterprise membutuhkan pemahaman tentang protokol transport dan keamanan.

#### 1. Transports: Stdio vs SSE (HTTP)

MCP mendukung dua jenis transport utama untuk komunikasi JSON-RPC 2.0:
- **Stdio Transport**:
  - Berjalan secara lokal. Host menjalankan sub-proses server dan berkomunikasi melalui `stdin` dan `stdout`.
  - Sangat aman karena tidak ada port jaringan yang terbuka.
  - Cocok untuk asisten lokal (seperti Cursor/VS Code plugin).
- **SSE (Server-Sent Events) / HTTP Transport**:
  - Menghubungkan client dan server yang berada di mesin/cloud berbeda secara remote.
  - Client mengirimkan perintah JSON-RPC melalui request **HTTP POST** biasa.
  - Server mengembalikan respon secara realtime menggunakan koneksi stream **SSE (Server-Sent Events)** satu arah.

#### 2. Streamable HTTP (Stateless MCP)
Pada arsitektur cloud serverless (seperti AWS Lambda atau Vercel Edge), menjaga koneksi SSE persisten adalah hal yang mahal dan tidak efisien.
- **Streamable HTTP (Stateless HTTP)** diperkenalkan di 2026 sebagai transport alternatif.
- Koneksi tidak perlu dibiarkan terbuka (stateless). Setiap request request-response berjalan mandiri, mirip REST API biasa, tetapi tetap mematuhi skema pesan JSON-RPC MCP.
- Ini mempermudah load-balancing dan deployment multi-tenant di cluster Kubernetes.

#### 3. Model Keamanan & Autentikasi (Security & Auth Model)
Karena MCP server bisa memanggil sistem sensitif (database, cloud terminal), model keamanan sangat diperketat:
- **Transport Layer Security (TLS)**: Semua koneksi remote SSE/HTTP wajib menggunakan HTTPS.
- **Token Delegation**: Saat client memanggil server remote, ia menyertakan token otorisasi OAuth/JWT di header HTTP. Server MCP harus melakukan verifikasi token secara terpisah sebelum mengeksekusi tool.
- **Sandboxing**: Host bertanggung jawab menjalankan local stdio server di dalam environment terisolasi (seperti gVisor atau Docker container ringkas) untuk menghindari malware mengeksploitasi filesystem lokal melalui tool filesystem.

---

### ⚠️ Jebakan Umum

**Jebakan 1: "MCP server = web server biasa"**
Tidak. Secara historis, MCP server berkomunikasi via **stdio** (stdin/stdout) lokal. Mengonversinya ke server remote membutuhkan konfigurasi transport SSE/HTTP atau Streamable HTTP, lengkap dengan manajemen SSL/TLS dan authentication layer.

**Jebakan 2: "Semua MCP server aman untuk dipakai"**
Hati-hati. Karena ekosistem MCP terdesentralisasi, siapa pun bisa mempublikasikan MCP server. Selalu verifikasi sumber server sebelum menggunakannya — terutama yang membutuhkan akses ke credential atau data sensitif. Sandboxing stdio server lokal adalah mitigasi wajib.

**Jebakan 3: "A2A berarti agent harus saling melihat internal state"**
Justru sebaliknya. Salah satu prinsip desain A2A adalah **opacity** — agent bisa berkolaborasi dan mendelegasikan tugas tanpa perlu mengekspos memori internal, konfigurasi tools, atau prompt system mereka ke agent lain. Mereka berkomunikasi murni lewat penugasan Task dan pengembalian Artifact.

---

### 🧩 Latihan

**Level 1 — Recall:**
Jelaskan perbedaan antara MCP Stdio transport dan SSE/HTTP transport. Kapan kamu harus memilih menggunakan masing-masing transport?

**Level 2 — Aplikasi:**
Modifikasi FastMCP server cuaca di atas (kamu bisa menulis dan menjalankannya secara lokal di komputermu dengan perintah `uv run mcp_cuaca_server.py`). Tambahkan tool baru `rekomendasi_wisata(kota)` yang memanfaatkan tool `cek_cuaca` secara internal untuk memberikan daftar aktivitas outdoor yang disarankan.

**Level 3 — Desain:**
Jelaskan bagaimana **Streamable HTTP** mengatasi keterbatasan koneksi persisten Server-Sent Events (SSE) pada arsitektur cloud serverless. Rancang skema autentikasi menggunakan OAuth2 untuk mengamankan MCP server remote yang dapat diakses oleh agent dari luar corporate network.

---

### 📝 Rangkuman

| Konsep | Inti Pemahaman |
|--------|---------------|
| **Masalah $N \times M$** | Tanpa standar, integrasi setiap agent ke setiap tool menghasilkan puluhan baris kode kustom yang rapuh. |
| **MCP** | Model Context Protocol — standardisasi open-source hubungan agent-to-tool di bawah Agentic AI Foundation. |
| **Tiga Primitif MCP** | **Tools** (aksi aktif), **Resources** (data pasif yang bisa dibaca), dan **Prompts** (template instruksi terstruktur). |
| **MCP Transports** | Local stdio (cepat/aman) vs remote SSE/HTTP (fleksibel/remote) vs Streamable HTTP (stateless/serverless). |
| **Keamanan MCP** | Wajib menggunakan HTTPS, verifikasi token (OAuth/JWT), dan sandboxing process untuk server filesystem lokal. |
| **A2A** | Agent2Agent — standardisasi kolaborasi agent-to-agent menggunakan Agent Cards, Tasks, dan Artifacts. |

> **Takeaway utama**: MCP dan A2A adalah "USB-C" bagi dunia AI Agentic. Mereka memisahkan kekhawatiran integrasi API kustom, memungkinkan tools dikembangkan sekali dan dikonsumsi oleh model mana pun secara instan dan aman di 2026.

---

**Selanjutnya → Modul 4.3: Agent di Produksi** — Membangun agent itu mudah. Menjalankannya secara andal di produksi? Itu tantangan sebenarnya.

