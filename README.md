<div align="center">

```
 ____   _   _   _   _    _   ____       _     ____      _     _____   _   _   ___
/ ___| / \ | | | | / \  | | / ___|     / \   / ___|    / \   |_   _|| | | | |_ _|
\___ \/ _ \| |_| |/ _ \ | || |        / _ \  \___ \   / _ \    | |  | |_| |  | |
 ___) / ___ \  _  / ___ \| || |___ _ / ___ \  ___) | / ___ \   | |  |  _  |  | |
|____/_/   \_|_| |_|   |_|_| \____|(_)_/  \_\|____/ /_/   \_\  |_|  |_| |_| |___|
```

**Telemedicine infrastructure for rural India**

![Status](https://img.shields.io/badge/status-live-4ade80?style=flat-square)
![Frontend](https://img.shields.io/badge/frontend-netlify-00C7B7?style=flat-square&logo=netlify&logoColor=white)
![Backend](https://img.shields.io/badge/backend-render-46E3B7?style=flat-square&logo=render&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/flask-2.0+-000000?style=flat-square&logo=flask&logoColor=white)

[Live Demo](https://saharasaathi.netlify.app) &nbsp;·&nbsp; [Backend Repo](https://github.com/qwertyuiopas17/ram) &nbsp;·&nbsp; [Frontend Repo](https://github.com/qwertyuiopas17/master)

</div>

---

## Overview

Sahara Saathi is a full-stack telemedicine platform built to close the healthcare accessibility gap in rural India. It connects patients with doctors, community health workers, and pharmacies through a single integrated system — enabling remote consultations, digital prescriptions, medicine delivery, and emergency response, accessible in over 10 Indian languages.

The platform is built across two repositories: a **Python/Flask backend** deployed on Render, and a **Vanilla JS/HTML frontend** deployed on Netlify.

---

## Table of Contents

- [Features](#features)
- [User Roles](#user-roles)
- [Cross-Role Workflows](#cross-role-workflows)
- [AI & Integrations](#ai--integrations)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Demo Credentials](#demo-credentials)
- [Architecture](#architecture)

---

## Features

| Feature | Description |
|---|---|
| Live Video Consultations | Real-time doctor–patient video calls via Agora RTC |
| AI Health Assistant | Multilingual chatbot with custom NLU for symptom triage and navigation |
| Digital Prescriptions | Doctors issue prescriptions that instantly sync to patient records and pharmacies |
| SOS Emergency Alerts | Geolocation-based SOS push notifications to the nearest active Saathi |
| QR Health Records | Portable patient health profile encoded as a personal QR code |
| Medicine Reminders | Push notification reminders with in-notification "Mark as Taken" action |
| Multilingual Voice Input | Bhashini-powered ASR/NMT/TTS pipeline supporting Hindi, Punjabi, and English |
| PWA Support | Installable as a Progressive Web App on any mobile device |
| Pharmacy Integration | End-to-end medicine ordering, inventory management, and delivery tracking |
| Admin Dashboard | Platform-wide analytics, user management, and grievance resolution |

---

## User Roles

The platform has five independent role-based dashboards. Each role has a separate login and a completely different interface scoped to its responsibilities.

---

### Patient

New patients self-register through the platform. On successful registration, the system generates a unique **Patient ID** with the prefix `PAT` (e.g. `PAT000001`). This ID is used for all subsequent logins — there is no email-based login.

**Registration fields:** Full Name · Age · Gender · Village/Area · District · Password

**Dashboard sections:**

**Home**
- Quick-access cards for all major features
- AI Health Chatbot — handles symptom queries, appointment booking, prescription summaries, medicine searches, and app navigation in multiple Indian languages
- Scan Medicine — camera-based medicine identifier

**Appointments**
- View upcoming and past appointments
- Book a new appointment by selecting specialty, doctor, and preferred time slot
- Join Live Demo Call — enters the shared video call channel with the assigned doctor

**Records**
- Complete digital health record: diagnoses, prescriptions, visit history
- Personal QR Code encoding the full health profile — scannable by any doctor for instant access

**Buy Medicine**
- Browse registered pharmacies
- Place medicine orders with home delivery
- Track order status on an integrated map view

**SOS**
- A single-tap emergency button that pushes a real-time alert to all logged-in Saathis within the nearest geographic radius, along with the patient's location

---

### Doctor

Doctors are registered by the Admin and assigned a Doctor ID (e.g. `DOC001`).

**Dashboard sections:**

**Home**
- Overview of today's consultations and upcoming appointments
- Quick-access to prescription upload

**Appointments**
- Full list of patient appointments with name, time, and status
- Join Demo Call — connects to the shared Agora channel for a live video session with the patient (both parties must be logged in simultaneously)

**Upload Rx**
- Select a patient by their `PAT...` ID
- Enter diagnosis, medications, dosages, and timing instructions
- On submission, the prescription is instantly pushed to the patient's Records and to all pharmacies in the patient's area

**Earnings**
- Consultation-by-consultation breakdown of payments received
- Tracks paid and pending statuses with a running total

---

### Saathi (Community Health Worker)

Saathis are grassroots health workers who bridge the platform and rural patients. Registered by the Admin, they operate from a field-facing dashboard.

**Dashboard sections:**

**Home / Task View**
- Pending tasks assigned to the Saathi
- Real-time SOS notifications — when a patient triggers SOS, all logged-in Saathis within the closest radius receive a push notification with the patient's location

**Log Activity**
- Submit field activity reports: patient onboarding, health checks, home visits
- Upload photo verification and set completion status
- Completed activities feed directly into earnings

**Training**
- Platform usage guides designed for low digital literacy
- Modules on assisting patients with app features

**Earnings**
- Per-activity income breakdown and monthly total

---

### Store (Pharmacy)

Pharmacies are registered by the Admin and assigned a Pharmacy ID (e.g. `PHAR001`).

**Dashboard sections:**

**Home**
- Pending order count and low-stock alerts at a glance

**Inventory**
- Full medicine catalog with current stock levels
- Update quantities, add new items, and view low-stock warnings

**Orders**
- All incoming patient orders with item details, quantities, and delivery addresses
- Map-based delivery route visualization for agents
- Update order status: Processing → Shipped → Delivered

**Prescriptions**
- All digital prescriptions issued by doctors for patients in the pharmacy's area
- View prescription details and convert directly to a medicine order

---

### Admin

The Admin has unrestricted access to all platform data and management tools.

**Dashboard sections:**

**Analytics**
- Active users, new registrations, daily consultations
- Medicine orders placed, disease trend tracking
- Chatbot and conversation volume metrics

**User Management**
- Full directory of Doctors, Saathis, and Pharmacies
- Verify, suspend, or remove accounts
- Approve new provider registrations

**Grievances**
- Support ticket queue with priority levels (High / Medium / Low)
- Resolve, close, or escalate complaints

---

## Cross-Role Workflows

These features involve simultaneous activity across two roles.

**Video Consultation (Doctor + Patient)**

```
Patient logs in  →  Appointments  →  "Join Demo Call"
Doctor logs in   →  Appointments  →  "Join Demo Call"
          └──── Both enter shared Agora RTC channel ────┘
```

Both parties must be logged in. The call is peer-to-peer through Agora's infrastructure — no third-party meeting link is needed.

**Digital Prescription Flow (Doctor → Patient → Pharmacy)**

```
Doctor: Upload Rx  →  selects PAT ID  →  submits prescription
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
   Patient: Records tab               Store: Prescriptions tab
   (instant sync)                     (ready to process as order)
```

**SOS Alert (Patient → Saathi)**

```
Patient presses SOS button
         │
         ▼
Backend calculates geolocation radius
         │
         ▼
All logged-in Saathis within range receive
push notification with patient location
```

---

## AI & Integrations

### AI Health Assistant

The chatbot is built on a custom NLU pipeline — not a generic API wrapper.

- **NLU Processor** (`nlu_processor.py`) — `scikit-learn`-based intent classifier that detects intents like `appointment_booking`, `symptom_triage`, `emergency_assistance`, `find_medicine`, and more. Uses `langdetect` for automatic language detection.
- **Response Generator** (`ko.py`) — rule-based multilingual response maps for English, Hindi, and Punjabi. Falls back to Groq LLM (`llama-3.3-70b-versatile`) for complex queries.
- **Conversation Memory** (`conversation_memory.py`) — persistent cross-session user profiles including medicine reminder schedules.
- **Medical Safety Guard** — any request resembling a diagnosis query is intercepted and redirected to doctor booking before the LLM is ever called.

### Bhashini (MeitY ULCA)

India's national AI language platform powers the full voice input pipeline:

- **ASR** — converts spoken audio (WebM → WAV) to text in the user's language
- **NMT** — translates from Hindi/Punjabi to English before processing, then back for the response
- **TTS** — converts the final text response back to spoken audio in the user's language

The pipeline runs on `call_bhashini_pipeline()` with an LRU-cached config lookup per session.

### Agora RTC

- Token-based channel authentication via `agora-token-builder`
- Tokens are generated server-side with a 1-hour expiry
- Powers the Live Demo Call between doctors and patients

### Web Push Notifications (VAPID)

- `pywebpush` + `py-vapid` send push notifications to subscribed browsers
- `APScheduler` runs a background job every 60 seconds to check for due medicine reminders
- Notifications include a **"Mark as Taken"** inline action button
- SOS alerts are dispatched through the same push pipeline

### Groq API

Two separate Groq clients are initialized on startup:

| Key | Model | Purpose |
|---|---|---|
| `GROQ_LLM_KEY` | `llama-3.3-70b-versatile` | Chatbot LLM responses, prescription structuring |
| `GROQ_WHISPER_KEY` | `whisper-large-v3` | Voice transcription fallback |

---

## Tech Stack

**Frontend**

| | |
|---|---|
| HTML5 / CSS3 / Vanilla JS | Core structure and logic |
| Progressive Web App | Installable, service worker caching |
| Agora Web SDK | Video call interface |
| Netlify | Hosting + `_redirects` for SPA routing |

**Backend**

| | |
|---|---|
| Python 3.9+ | Core language |
| Flask | Web framework and REST API |
| Flask-SocketIO + Gevent | WebSocket / real-time support |
| Flask-SQLAlchemy | ORM |
| PostgreSQL | Production database (Render) |
| SQLite | Local development database |
| Gunicorn | WSGI server |
| Render | Backend hosting |

**AI / ML**

| | |
|---|---|
| Groq | LLM inference and Whisper transcription |
| scikit-learn | NLU intent classification |
| numpy / pandas | Data processing |
| langdetect | Language detection |
| pydub / audioop-lts | Audio format conversion |

**Services**

| | |
|---|---|
| Agora RTC | Video infrastructure |
| Bhashini ULCA | Indian language ASR / NMT / TTS |
| pywebpush / py-vapid | Web push notifications |
| APScheduler | Background job scheduling |
| bcrypt | Password hashing |

---

## Repository Structure

**Frontend — [`qwertyuiopas17/master`](https://github.com/qwertyuiopas17/master)**

```
├── index.html                        # Landing page
├── patient.html                      # Patient dashboard
├── doctor.html                       # Doctor dashboard
├── saathi.html                       # Saathi dashboard
├── store.html                        # Pharmacy dashboard
├── admin.html                        # Admin control panel
├── read_me.html                      # In-app usage guide
├── test_video.html                   # Video call test page
├── example_chatbot_integration.html  # Chatbot integration reference
├── frontend_button_handler.js        # AI chatbot action dispatcher
├── manifest.json                     # PWA manifest
├── service-worker.js / sw.js         # Service workers for offline support
├── icons.json                        # Icon definitions
├── icons/                            # PWA icon assets
├── static/                           # CSS, JS, and image assets
└── _redirects                        # Netlify SPA routing rules
```

**Backend — [`qwertyuiopas17/ram`](https://github.com/qwertyuiopas17/ram)**

```
├── chatbot.py                    # Main Flask app — all routes, auth, WebSocket handlers
├── ko.py                         # ProgressiveResponseGenerator — multilingual intent maps
├── nlu_processor.py              # ProgressiveNLUProcessor — intent classification
├── api_ollama_integration.py     # Groq API client wrapper
├── conversation_memory.py        # Cross-session user profiles and reminder state
├── enhanced_database_models.py   # SQLAlchemy models (User, Doctor, Pharmacy, etc.)
└── requirements.txt              # Python dependencies
```

---

## Local Setup

### Prerequisites

- Python 3.9+
- pip
- PostgreSQL (or SQLite for local dev — no setup needed)

### Backend

```bash
git clone https://github.com/qwertyuiopas17/ram.git
cd ram

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Add your environment variables (see section below)
cp .env.example .env

python chatbot.py
```

Server runs at `http://localhost:5000`.

### Frontend

```bash
git clone https://github.com/qwertyuiopas17/master.git
cd master

# Serve with any static file server
npx serve .
# or use VS Code Live Server
```

By default the frontend points to the deployed backend. To use your local backend, update the API base URL in the JS files from the Render URL to `http://localhost:5000`.

---

## Environment Variables

Create a `.env` file in the backend root:

```env
# Database
DATABASE_URL=postgresql://user:password@host:port/dbname

# Flask
SECRET_KEY=your-secret-key

# Groq
GROQ_LLM_KEY=your-groq-llm-key
GROQ_WHISPER_KEY=your-groq-whisper-key

# Agora
AGORA_APP_ID=your-agora-app-id
AGORA_APP_CERTIFICATE=your-agora-certificate

# VAPID (Web Push) — required, app will not start without these
VAPID_PUBLIC_KEY=your-vapid-public-key
VAPID_PRIVATE_KEY=your-vapid-private-key

# Bhashini
BHASHINI_USER_ID=your-bhashini-user-id
BHASHINI_ULCA_API_KEY=your-bhashini-ulca-api-key
```

> VAPID keys are mandatory — the server exits immediately on startup if they are missing. Generate a pair at [vapidkeys.com](https://vapidkeys.com) or via `py-vapid`.

---

## Demo Credentials

Pre-seeded accounts for testing:

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |
| Doctor | `DOC001` | `password123` |
| Saathi | `SAATHI001` | `password123` |
| Pharmacy | `PHAR001` | `password123` |
| Patient | `PAT001` | `password123` |

**Registering a new patient**

Click Register on the landing page, select the Patient role, and fill in the form. The system generates a unique `PAT...` ID on success — save it, as it is the login username.

**Testing the video call**

Open two browser tabs. Log in as `DOC001` in one and `PAT001` in the other. Both go to Appointments and click "Join Demo Call". The call connects when both are in the channel simultaneously.

**Testing SOS**

Log in as `SAATHI001` in one tab. In another tab, log in as `PAT001` and press the SOS button. The Saathi tab receives a push notification with the patient's location.

---

## Architecture

```
                     ┌──────────────────────────────┐
                     │      Netlify (Frontend)       │
                     │  index · patient · doctor     │
                     │  saathi · store · admin       │
                     │  PWA  ·  Agora Web SDK        │
                     └──────────────┬───────────────┘
                                    │
                          HTTP REST + WebSocket
                                    │
                     ┌──────────────▼───────────────┐
                     │       Render (Backend)        │
                     │    Flask + Gunicorn           │
                     │    Flask-SocketIO + Gevent    │
                     │                               │
                     │  ┌─────────┐  ┌───────────┐  │
                     │  │   NLU   │  │ Response  │  │
                     │  │Processor│  │ Generator │  │
                     │  └────┬────┘  └─────┬─────┘  │
                     │       └──────┬───────┘        │
                     │         ┌────▼────┐            │
                     │         │  Groq   │            │
                     │         │  LLM    │            │
                     │         └─────────┘            │
                     │                               │
                     │  PostgreSQL   Bhashini ULCA   │
                     │  Agora RTC    VAPID Push      │
                     └──────────────────────────────┘
```

---

<div align="center">

![visitors](https://img.shields.io/badge/built%20with-Python%20%26%20Vanilla%20JS-informational?style=flat-square)

[saharasaathi.netlify.app](https://saharasaathi.netlify.app)

</div>
