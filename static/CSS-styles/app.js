document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("uploadArea");
    const fileInput = document.getElementById("files");
    const fileList = document.getElementById("fileList");
    const log = document.getElementById("log");
    const startBtn = document.getElementById("startBtn");
    const resetBtn = document.getElementById("resetBtn");
    const singleDocInput = document.getElementById('singleDoc');

    // Drag & Drop
    uploadArea.addEventListener("click", () => fileInput.click());
    uploadArea.addEventListener("dragover", (e) => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
    uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("drag-over");
        fileInput.files = e.dataTransfer.files;
        updateFileList();
    });

    fileInput.addEventListener("change", updateFileList);
    resetBtn.addEventListener("click", reset);
    startBtn.addEventListener("click", start);

    function updateFileList() {
        fileList.innerHTML = "";
        if (fileInput.files.length === 0) return;

        const container = document.createElement("div");
        container.className = "file-list";

        Array.from(fileInput.files).forEach((file, index) => {
            const item = document.createElement("div");
            item.className = "file-item success";
            item.innerHTML = `
                <div class="file-item-name">
                    <span>✓</span> <span>${file.name}</span>
                    <span style="color:var(--text-secondary);font-size:13px;">(${(file.size/1024/1024).toFixed(2)} MB)</span>
                </div>
                <div class="file-item-remove" data-index="${index}">Remove</div>
            `;
            container.appendChild(item);
        });

        fileList.appendChild(container);
        
        // Add click handlers for remove buttons
        document.querySelectorAll('.file-item-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const dt = new DataTransfer();
                const files = fileInput.files;
                const idxToRemove = parseInt(e.target.dataset.index);
                for (let i = 0; i < files.length; i++) {
                    if (i !== idxToRemove) dt.items.add(files[i]);
                }
                fileInput.files = dt.files;
                updateFileList();
            });
        });
    }

    async function start() {
        if (!fileInput.files.length) {
            log.innerHTML = '<div style="color:#ef4444;text-align:center;padding:20px;">⚠️ Select at least one file.</div>';
            return;
        }

        startBtn.disabled = true;
        startBtn.innerText = 'Processing...';
        log.innerHTML = `
            <div class="progress-container">
                <div class="progress-bar"><div class="progress-fill" style="width:100%;animation:shimmer 2s infinite;"></div></div>
                <div class="progress-status">Transcribing... Please wait.</div>
            </div>`;

        const formData = new FormData();
        formData.append("prompt", document.getElementById("prompt").value);
        formData.append("single_doc", singleDocInput.checked);
        for (const file of fileInput.files) formData.append("files", file);

        try {
            const response = await fetch("/transcribe", { method: "POST", body: formData });
            if (!response.ok) throw new Error(response.statusText);
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            log.innerHTML = `<div style="color:#ef4444;text-align:center;padding:20px;">❌ Error: ${error.message}</div>`;
        } finally {
            startBtn.disabled = false;
            startBtn.innerText = 'Start Transcription';
        }
    }

    function displayResults(data) {
        let html = '';
        if (data.google_doc_id) {
            html += `
                <div class="result-card" style="background:rgba(16,185,129,0.1);border-color:rgba(16,185,129,0.3);">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-weight:600;color:#10b981;">✨ All Files Merged</div>
                            <div style="color:var(--text-secondary);font-size:0.875rem;">Saved to Master Google Doc</div>
                        </div>
                        <a href="https://docs.google.com/document/d/${data.google_doc_id}/edit" target="_blank" 
                           class="copy-btn" style="text-decoration:none;background:#10b981;color:white;">Open Master Doc →</a>
                    </div>
                </div>`;
        }

        if (data.results && data.results.length) {
            data.results.forEach((res, i) => {
                let link = (!data.google_doc_id && res.individual_doc_id) ? 
                    `<a href="https://docs.google.com/document/d/${res.individual_doc_id}/edit" target="_blank" class="copy-btn" style="text-decoration:none;margin-right:8px;background:#3b82f6;color:white;">Open Doc ↗</a>` : '';
                
                html += `
                    <div class="result-card">
                        <div class="result-card-header">
                            <div class="result-card-title">${res.filename}</div>
                            <div>${link}<button class="copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('res-${i}').innerText)">📋 Copy</button></div>
                        </div>
                        <div class="result-text" id="res-${i}">${res.text || "⚠️ No text."}</div>
                    </div>`;
            });
        }
        log.innerHTML = html;
    }

    function reset() {
        fileInput.value = "";
        document.getElementById("prompt").value = "";
        singleDocInput.checked = true;
        fileList.innerHTML = "";
        log.innerHTML = '<span class="empty-state">Results will appear here</span>';
    }
});