document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("user-input");
    const messages = document.getElementById("chat-messages");
    const welcome = document.getElementById("welcome-card");
    const sendButton = document.getElementById("send-button");
    const sourcesList = document.getElementById("sources-list");
    const statPdf = document.getElementById("stat-pdf");
    const statApi = document.getElementById("stat-api");
    let sending = false;

    const setStatus = (dotId, labelId, state, label) => {
        document.getElementById(dotId).className = `status-dot ${state}`;
        document.getElementById(labelId).textContent = label;
    };

    async function checkStatus() {
        try {
            const response = await fetch("/api/status", {headers: {"Accept": "application/json"}});
            if (!response.ok) throw new Error("status unavailable");
            const status = await response.json();
            const vector = status.vectorstore;
            setStatus("db-dot", "db-label", vector.status === "ready" ? "online" : "warning",
                vector.status === "ready"
                    ? `İlaç Bilgi Arşivi Hazır (${vector.unique_medicines || 0} ilaç, ${vector.pdf_count || 0} belge)`
                    : "İlaç Bilgi Arşivi Hazırlanıyor");
            const llmReady = status.llm.status === "ready" || status.llm.status === "configured";
            setStatus("ai-dot", "ai-label", llmReady ? "online" : "warning",
                llmReady ? "Akıllı Yanıt Sistemi Hazır" : "Akıllı Yanıt Sistemi Sınırlı");
            setStatus("api-dot", "api-label", status.medicine_api.status === "online" ? "online" : "offline",
                status.medicine_api.status === "online" ? "İlaç Bilgi Servisi Aktif" : "İlaç Bilgi Servisi Kapalı");
        } catch (_) {
            setStatus("db-dot", "db-label", "offline", "İlaç Bilgi Arşivine Ulaşılamıyor");
            setStatus("ai-dot", "ai-label", "offline", "Akıllı Yanıt Sistemine Ulaşılamıyor");
            setStatus("api-dot", "api-label", "offline", "İlaç Bilgi Servisine Ulaşılamıyor");
        }
    }

    function addMessage(sender, text, loading = false) {
        if (!loading) document.querySelector(".message.assistant.loading")?.remove();
        const wrapper = document.createElement("div");
        wrapper.className = `message ${sender}${loading ? " loading" : ""}`;
        const title = document.createElement("div");
        title.className = "message-sender";
        title.textContent = sender === "user" ? "Siz" : "Tıbbi Asistan";
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        if (loading) {
            const dots = document.createElement("div");
            dots.className = "typing-dots";
            for (let i = 0; i < 3; i += 1) dots.appendChild(Object.assign(document.createElement("div"), {className: "dot"}));
            bubble.appendChild(dots);
        } else {
            bubble.textContent = text;
            bubble.style.whiteSpace = "pre-wrap";
        }
        wrapper.append(title, bubble);
        messages.appendChild(wrapper);
        messages.scrollTop = messages.scrollHeight;
    }

    function addSuggestedQuestions(questions = []) {
        if (!Array.isArray(questions) || !questions.length) return;
        const container = document.createElement("div");
        container.className = "suggested-follow-ups";
        questions.forEach(question => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "suggestion-chip";
            button.textContent = String(question);
            button.addEventListener("click", () => {
                input.value = String(question);
                input.focus();
            });
            container.appendChild(button);
        });
        messages.appendChild(container);
    }

    const field = (label, value) => {
        const row = document.createElement("p");
        const strong = document.createElement("strong");
        strong.textContent = `${label}: `;
        row.append(strong, document.createTextNode(value ?? "-"));
        return row;
    };

    function updateSources(sources = [], stats = {}) {
        statPdf.textContent = String(stats.pdf_count || 0);
        statApi.textContent = String(stats.api_count || 0);
        sourcesList.replaceChildren();
        if (!sources.length) {
            const empty = document.createElement("div");
            empty.className = "empty-sources";
            empty.textContent = "Bu cevap için doğrulanmış kaynak kullanılmadı.";
            sourcesList.appendChild(empty);
            return;
        }
        sources.forEach(source => {
            const card = document.createElement("div");
            card.className = "source-card open";
            const header = document.createElement("div");
            header.className = "source-header";
            const filename = String(source.source || "").split(/[\\/]/).pop() || "Kaynak";
            header.textContent = filename;
            const body = document.createElement("div");
            body.className = "source-body";
            body.append(
                field("İlaç", source.drug_name),
                field("Belge türü", source.document_type),
                field("Sayfa", source.page),
                field("Skor", typeof source.score === "number" ? source.score.toFixed(4) : "-")
            );
            if (source.source_url) {
                try {
                    const url = new URL(source.source_url);
                    if (["http:", "https:"].includes(url.protocol)) {
                        const link = document.createElement("a");
                        link.href = url.href;
                        link.target = "_blank";
                        link.rel = "noopener noreferrer";
                        link.textContent = "Resmî kaynağı aç";
                        body.appendChild(link);
                    }
                } catch (_) { /* güvenli olmayan/geçersiz URL gösterilmez */ }
            }
            card.append(header, body);
            header.addEventListener("click", () => card.classList.toggle("open"));
            sourcesList.appendChild(card);
        });
    }

    async function sendQuestion(rawQuestion) {
        const question = rawQuestion.trim();
        if (!question || sending) return;
        sending = true;
        input.disabled = true;
        sendButton.disabled = true;
        if (welcome) welcome.style.display = "none";
        addMessage("user", question);
        addMessage("assistant", "", true);
        try {
            const response = await fetch("/api/ask", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question})
            });
            let data;
            try { data = await response.json(); } catch (_) { data = {}; }
            if (!response.ok) throw new Error(data.detail || "İstek işlenemedi.");
            addMessage("assistant", data.answer || "Sunucu boş bir cevap döndürdü.");
            addSuggestedQuestions(data.suggested_questions);
            updateSources(data.sources, data.retrieval_stats);
        } catch (error) {
            const offline = error instanceof TypeError;
            addMessage("assistant", offline ? "Sunucuya ulaşılamadı. Bağlantınızı kontrol edip tekrar deneyin." : `İstek tamamlanamadı: ${error.message}`);
        } finally {
            sending = false;
            input.disabled = false;
            sendButton.disabled = false;
            input.value = "";
            input.focus();
        }
    }

    form.addEventListener("submit", event => { event.preventDefault(); sendQuestion(input.value); });
    input.addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            form.requestSubmit();
        }
    });
    window.sendSuggestion = sendQuestion;
    checkStatus();
    setInterval(checkStatus, 15000);
});
