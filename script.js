// ===============================================
// 1. GLOBAL INTERACTIVE FUNCTIONS (Must be defined globally for HTML onclick access)
// ===============================================

// --- GPS Capture Simulation (Global) ---
document.getElementById("login-form").addEventListener("submit", function (e) {
    e.preventDefault();

    const name = document.getElementById("farmer-name").value;

    document.getElementById("login-page").style.display = "none";
    document.getElementById("app").style.display = "block";

    document.getElementById("farmer-display-name").textContent = name;
});
window.captureGPS = function() {
    const gpsStatusElement = document.getElementById('gps-status');
    const captureBtn = document.getElementById('capture-gps-btn');
    
    // 1. Show 'Capturing' state
    gpsStatusElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Capturing...';
    captureBtn.disabled = true;
    
    // 2. Simulate delay for GPS lock
    setTimeout(() => {
        // Simulated GPS Data
        const lat = (Math.random() * 5 + 18).toFixed(4);
        const lon = (Math.random() * 5 + 72).toFixed(4);
        
        // 3. Update status to Success
        gpsStatusElement.innerHTML = `<i class="fas fa-check-circle"></i> ${lat}°N, ${lon}°E`;
        captureBtn.textContent = 'Location Captured';
        captureBtn.style.backgroundColor = '#28a745'; // Green for success
        
        alert(`Success! Field location captured: ${lat}°N, ${lon}°E. This data is now stored for hyperlocal weather and risk analysis.`);
    }, 1500); // 1.5 second delay
}

// --- Map Filter Simulation (Global) ---
window.updateMapOverlay = function(value) {
    const mapArea = document.querySelector('.map-area-placeholder');
    const forecastBox = document.querySelector('.map-overlay');
    const riskMarker = document.querySelector('.map-marker.risk');

    // Reset Marker state
    riskMarker.style.display = 'none';

    if (value === 'Pest') {
        mapArea.style.border = '2px solid #d32f2f';
        forecastBox.innerHTML = '<p class="forecast-title" style="color: #d32f2f;">Overlay Active: HIGH PEST RISK</p><p>Showing zones needing immediate treatment.</p>';
        riskMarker.style.display = 'inline-block';
    } else if (value === 'Weather') {
        mapArea.style.border = '2px solid #007bff';
        forecastBox.innerHTML = '<p class="forecast-title" style="color: #007bff;">Overlay Active: RAINFALL RADAR</p><p>Showing rain probability for the next 24 hours.</p>';
    } else if (value === 'Moisture') {
        mapArea.style.border = '2px solid #ff9800';
        forecastBox.innerHTML = '<p class="forecast-title" style="color: #ff9800;">Overlay Active: SOIL MOISTURE</p><p>Plot B indicates 30% low moisture. Advisory: Begin irrigation cycle.</p>';
    } else {
        mapArea.style.border = '2px solid #388e3c';
        forecastBox.innerHTML = '<p class="forecast-title" style="color: #008000;">Hyperlocal Forecast: 7-Day View</p><p>Status: Normal Field View</p>';
    }
}

// --- Quick Query Handler (Global) ---
window.sendQuickQuery = function(query) {
    const userInput = document.getElementById('user-input');
    userInput.value = query;
    if (typeof sendMessage === 'function') {
        sendMessage();
    }
}

// --- Voice Input Handler (Global) ---
window.handleVoiceInput = function() {
    const micButton = document.getElementById('mic-button');
    micButton.classList.add('recording');
    micButton.innerHTML = '<i class="fas fa-stop"></i>';
    alert("Simulating Voice Input: Speak your query now.");
    
    setTimeout(() => {
        micButton.classList.remove('recording');
        micButton.innerHTML = '<i class="fas fa-microphone"></i>';
        const simulatedVoiceText = "Show scheme details for crop insurance."; 
        const userInput = document.getElementById('user-input');
        userInput.value = simulatedVoiceText;
        if (typeof sendMessage === 'function') {
             sendMessage();
        }
    }, 1500); 
}


// ===============================================
// 2. DOMContentLoaded (Page Setup, Tab Switching, and Chat Logic)
// ===============================================

document.addEventListener('DOMContentLoaded', () => {
    
    // --- Initial Variable Setup ---
    const navLinks = document.querySelectorAll('.nav-tabs a');
    const dashboardView = document.getElementById('dashboard');
    const chatSidebar = document.getElementById('chat');
    const mapView = document.getElementById('map'); 
    const allViews = [dashboardView, chatSidebar, mapView];

    dashboardView.style.display = 'block';

    // --- Tab Switching and Background Logic ---
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // 1. Handle View Switching (Visibility)
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            allViews.forEach(v => {
                v.style.display = 'none';
                v.classList.remove('active-view');
            });
            
            const targetId = link.getAttribute('href').substring(1);
            const targetView = document.getElementById(targetId);

            if (targetView) {
                targetView.style.display = 'block';
                targetView.classList.add('active-view');
            } else {
                alert(`Navigating to the ${targetId} view. In a full project, this would load the Data Explorer.`);
                dashboardView.style.display = 'block'; 
                dashboardView.classList.add('active-view');
            }

            // 2. Handle Background Change (The Core Feature)
            
            // Remove all specific background classes first
            document.body.classList.remove('agri-background', 'bg-weather', 'bg-chat'); 

            if (targetId === 'map') {
                // Apply specific class for the Map/Weather view
                document.body.classList.add('bg-weather'); 
            } else if (targetId === 'chat') {
                // Apply specific class for the Chat view
                document.body.classList.add('bg-chat'); 
            } else {
                // Apply the default class (Dashboard/Data)
                document.body.classList.add('agri-background');
            }
        });
    });

    // --- Chatbot Setup ---
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');

    userInput.disabled = false; 

    // Function to append a message to the chat
    function appendMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        
        if (sender === 'ai') {
            msgDiv.innerHTML = '<i class="fas fa-magic"></i> AI is thinking...';
            chatMessages.appendChild(msgDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            setTimeout(() => {
                msgDiv.innerHTML = text; 
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 1000); 
        } else {
            msgDiv.textContent = text;
            chatMessages.appendChild(msgDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    // Function to handle user input (text or simulated voice)
    window.sendMessage = function() { 
        const text = userInput.value.trim();
        if (text === '') return;

        appendMessage(text, 'user');
        userInput.value = '';

        setTimeout(() => {
            const aiResponse = generateSimulatedResponse(text);
            appendMessage(aiResponse, 'ai');
        }, 500); 
    }

    // Keypress listener for Enter/Return key
    // Keydown listener for Enter key (Note: switched to keydown for better multi-key detection)
userInput.addEventListener('keydown', (e) => {
    // Check if Enter key is pressed AND Ctrl or Cmd key is also pressed
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault(); // Prevents adding a new line
        sendMessage();
    }
    // If only Enter is pressed, it will act normally (add a new line)
});

    // --- Simulated AI Response Generation ---
    function generateSimulatedResponse(query) {
        const lowerQuery = query.toLowerCase();

        if (lowerQuery.includes('weather') || lowerQuery.includes('rain') || lowerQuery.includes('frost')) {
            return "Based on hyperlocal data, a mild **frost alert** is issued for your area tomorrow night. Please ensure your irrigation is managed to prevent crop damage. **Actionable Step:** Use the 'Field/Weather' tab for precise map details.";
        }
        if (lowerQuery.includes('price') || lowerQuery.includes('corn') || lowerQuery.includes('market')) {
            return "The current market price for **Corn** at the nearest mandi (Mandi 'X') is **₹ 22.50 per kg**. This is 3% higher than the state average. **Decision Support:** Recommend selling within the next 7 days as demand is peaking.";
        }
        if (lowerQuery.includes('pesticide') || lowerQuery.includes('disease') || lowerQuery.includes('cotton')) {
            return "For cotton crop protection against **Bollworm**, the recommended pesticide is **Product 'Y'**. Use a dosage of 1.5ml per liter of water. **Sustainability Guidance:** Apply only during calm wind conditions to optimize resource use.";
        }
        if (lowerQuery.includes('scheme') || lowerQuery.includes('insurance')) {
            return "The **Pradhan Mantri Fasal Bima Yojana (PMFBY) for crop insurance** is active. **Guidance:** You can apply via our integrated form. The last date for enrollment is the 30th of this month.";
        }
        
        return "I'm sorry, I couldn't find specific information for that query. Can you try asking about weather, market prices, or government schemes?";
    }
});