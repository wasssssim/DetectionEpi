// --- CONFIGURATION ---
const HOST = window.location.hostname || 'localhost';
const PORT = '8000';
const BACKEND = `http://${HOST}:${PORT}`;
const WS_URL = `ws://${HOST}:${PORT}/ws`;

// Liste des IDs caméras (Doit être identique à ton dictionnaire Python 'cameras')
const CAMERAS = ['cam_1', 'cam_2']; 

// --- ELEMENTS DOM ---
const grid = document.getElementById('monitor-grid');
const alertContainer = document.getElementById('history-list');
const toast = document.getElementById('toast');
const toastMsg = document.getElementById('toast-msg');

/**
 * 1. INITIALISATION DE LA GRILLE
 * Crée dynamiquement les cartes pour chaque caméra
 */
function initGrid() {
    if (!grid) return;
    grid.innerHTML = CAMERAS.map(id => `
        <div class="cam-card" id="card-${id}">
            <div class="cam-overlay">
                <span class="cam-badge">LIVE</span>
                <span class="cam-name">${id.toUpperCase()}</span>
            </div>
            <img class="cam-video" src="${BACKEND}/video/${id}" alt="Flux ${id}">
            <div class="alert-border"></div>
        </div>
    `).join('');
}

/**
 * 2. GESTION DES ALERTES VISUELLES (Le "Dessin" dynamique)
 * Fait clignoter la carte de la caméra concernée en rouge
 */
function triggerVisualAlert(camId) {
    const card = document.getElementById(`card-${camId}`);
    if (card) {
        card.classList.add('in-error');
        // On retire l'effet après 6 secondes
        setTimeout(() => card.classList.remove('in-error'), 6000);
    }
}

/**
 * 3. WEBSOCKET : RÉCEPTION EN TEMPS RÉEL
 */
function setupWebSocket() {
    const socket = new WebSocket(WS_URL);

    socket.onopen = () => console.log("✅ Connecté au serveur d'alertes");

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "ALERTE") {
            // Déclenche le "dessin" clignotant sur la grille
            triggerVisualAlert(data.camera);
            // Affiche la notification Toast
            showToast(`⚠️ INFRACTION : ${data.epi} sur ${data.camera.toUpperCase()}`);
            // Rafraîchit les KPI et l'historique
            updateUI();
        }
    };

    socket.onclose = () => {
        console.log("❌ Déconnecté. Reconnexion...");
        setTimeout(setupWebSocket, 3000);
    };
}

/**
 * 4. API : RÉCUPÉRATION DES STATS ET DE L'HISTORIQUE
 */
async function updateUI() {
    // Récupération des KPI (Aujourd'hui / Hier)
    try {
        const stats = await fetch(`${BACKEND}/stats`).then(r => r.json());
        document.getElementById('kpi-today').textContent = stats.today;
        document.getElementById('kpi-yesterday').textContent = stats.yesterday;
    } catch (e) { console.error("Erreur Stats:", e); }

    // Récupération de l'historique
    try {
        const history = await fetch(`${BACKEND}/history`).then(r => r.json());
        renderHistory(history);
    } catch (e) { console.error("Erreur History:", e); }
}

function renderHistory(items) {
    if (!alertContainer) return;
    document.getElementById('count-badge').textContent = `${items.length} alerte(s)`;

    if (items.length === 0) {
        alertContainer.innerHTML = '<div class="empty-state">Aucune infraction détectée</div>';
        return;
    }

    alertContainer.innerHTML = items.map(item => `
        <div class="history-card" onclick='openModal(${JSON.stringify(item)})'>
            <div class="card-date">
                <span class="day">${item.date.split(' ')[0]}</span>
                <span class="time">${item.date.split(' ')[1]}</span>
            </div>
            <div class="card-epi">
                <span class="epi-tag">${item.epi}</span>
            </div>
            <div class="card-thumb">
                <img src="${BACKEND}${item.photo}" alt="capture">
            </div>
        </div>
    `).join('');
}

/**
 * 5. MODALE ET TOAST
 */
function openModal(item) {
    document.getElementById('modal-img').src = `${BACKEND}${item.photo}`;
    document.getElementById('modal-id').innerHTML = `<strong>ID :</strong> #${item.id}`;
    document.getElementById('modal-date').innerHTML = `<strong>Horodatage :</strong> ${item.date}`;
    document.getElementById('modal-epi').innerHTML = `<strong>Infraction :</strong> ${item.epi}`;
    document.getElementById('modal').classList.add('open');
}

function closeModal() { document.getElementById('modal').classList.remove('open'); }

function showToast(msg) {
    toastMsg.textContent = msg;
    toast.classList.add('visible');
    setTimeout(() => toast.classList.remove('visible'), 5000);
}

// --- LANCEMENT ---
window.onload = () => {
    initGrid();
    setupWebSocket();
    updateUI();
    
    // Horloge temps réel
    setInterval(() => {
        const el = document.getElementById('clock');
        if (el) el.textContent = new Date().toLocaleTimeString('fr-FR');
    }, 1000);
};