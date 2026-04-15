document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = document.getElementById('chat-history');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const uploadBtn = document.getElementById('upload-btn');
    const imageUpload = document.getElementById('image-upload');
    const reportBtn = document.getElementById('generate-report');

    let currentNlpExtracts = [];
    let currentStructuralMatches = [];

    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    uploadBtn.addEventListener('click', () => {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            
            if (file.type.startsWith('image/')) {
                appendMessage('user', `Analyzing visual attachment: ${file.name} 📎`);
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/visual_match', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        let msg = `Found ${data.matches.length} globally similar structures:<br><br>`;
                        data.matches.forEach(m => {
                            msg += `• Target Patent: **${m.source_patent}** (Score: ${(m.similarity*100).toFixed(1)}%)<br>`;
                            msg += `<img src="${m.image_url}" alt="match" style="max-height: 150px; margin-top: 10px; border-radius: 8px; border: 1px solid var(--border-color);"><br><br>`;
                        });
                        currentStructuralMatches = data.matches;
                        appendMessage('ai', msg);
                    } else {
                        appendMessage('ai', `<span style="color: #ff6b6b;">Error: ${data.detail || 'Vision API failed'}</span>`);
                    }
                } catch (err) {
                    appendMessage('ai', `<span style="color: #ff6b6b;">Error: Vision backend offline</span>`);
                }
            } else if (file.type === 'application/pdf' || file.type === 'text/plain') {
                appendMessage('user', `Performing Freedom To Operate Analysis on: ${file.name} 📄`);
                
                const formData = new FormData();
                formData.append('file', file);
                const loadingId = appendLoadingMsg();

                try {
                    const response = await fetch('/api/analyze_document', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        currentNlpExtracts.push({
                            target: `Prior Art: ${data.closest_patents.join(', ')}`,
                            inquiry: `FTO Analysis for proposal: ${file.name}`,
                            exact_answer: data.analysis
                        });
                        
                        let sourcesHtml = data.closest_patents.length > 0 ? data.closest_patents.map(s => `<span style="font-size: 0.8rem; padding: 2px 6px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-right: 4px;">${s}</span>`).join('') : '<span style="font-size: 0.8rem; padding: 2px 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">No strict overlaps found</span>';

                        appendMessage('ai', `<div>${marked.parse(data.analysis)}</div><div style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 10px; display:flex; gap: 8px; align-items:center;"><strong style="font-size:0.8rem;">PRIOR ART FLAGGED:</strong> ${sourcesHtml}</div>`, true);
                    } else {
                         appendMessage('ai', `<span style="color: #ff6b6b;">Error: ${data.detail || 'Document Analysis failed'}</span>`);
                    }
                } catch (err) {
                    appendMessage('ai', `<span style="color: #ff6b6b;">Error: Backend offline</span>`);
                } finally {
                    removeLoadingMsg(loadingId);
                }
            }
            imageUpload.value = ''; 
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userInput = chatInput.value.trim();
        if (!userInput) return;

        appendMessage('user', userInput);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        const loadingId = appendLoadingMsg();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: userInput })
            });

            const data = await response.json();

            if (response.ok) {
                currentNlpExtracts.push({
                    target: data.sources.length > 0 ? data.sources.join(', ') : 'Global Knowledge',
                    inquiry: userInput,
                    exact_answer: data.answer
                });
                
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                     sourcesHtml = data.sources.map(s => `<span style="font-size: 0.8rem; padding: 2px 6px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-right: 4px;">${s}</span>`).join('');
                } else {
                     sourcesHtml = '<span style="font-size: 0.8rem; padding: 2px 6px; background: rgba(255,255,255,0.1); border-radius: 4px;">Global Context</span>';
                }

                appendMessage('ai', `<div>${marked.parse(data.answer)}</div><div style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 10px; display:flex; gap: 8px; align-items:center;"><strong style="font-size:0.8rem;">SOURCES:</strong> ${sourcesHtml}</div>`, true);
            } else {
                appendMessage('ai', `<span style="color: #ff6b6b;">Error: ${data.detail || 'Global RAG failed'}</span>`);
            }
        } catch (err) {
            appendMessage('ai', `<span style="color: #ff6b6b;">Error: Backend offline</span>`);
        } finally {
            removeLoadingMsg(loadingId);
        }
    });

    reportBtn.addEventListener('click', async () => {
        if (currentNlpExtracts.length === 0 && currentStructuralMatches.length === 0) {
            alert("No analysis performed yet to generate a report.");
            return;
        }

        const reportData = {
            nlp_extracts: currentNlpExtracts,
            structural_matches: currentStructuralMatches.map(m => ({
                source_patent: m.source_patent,
                image_filename: m.image_url.split('/').pop(),
                similarity: m.similarity
            }))
        };

        reportBtn.innerHTML = '🔄';
        try {
            const response = await fetch('/api/report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reportData)
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `smartcs_global_report_${new Date().toISOString().split('T')[0]}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
            } else {
                alert("Failed to generate PDF Report. Ensure backend is running and matches are valid.");
            }
        } catch (err) {
            alert("Server connection failed.");
        } finally {
            reportBtn.innerHTML = '📄';
        }
    });

    function appendMessage(sender, htmlContent, skipFormatting=false) {
        const div = document.createElement('div');
        div.className = `message ${sender}-msg`;
        
        let avatar = sender === 'ai' ? '🤖' : '👤';
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.innerHTML = avatar;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        
        if (skipFormatting) {
             contentDiv.innerHTML = htmlContent;
        } else {
             let formatted = htmlContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
             contentDiv.innerHTML = formatted;
        }

        div.appendChild(avatarDiv);
        div.appendChild(contentDiv);
        chatHistory.appendChild(div);
        
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    function appendLoadingMsg() {
        const id = 'loading-' + Date.now();
        const div = document.createElement('div');
        div.className = `message ai-msg`;
        div.id = id;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'avatar';
        avatarDiv.innerHTML = '🤖';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.innerHTML = '<div class="typing-indicator"><span>.</span><span>.</span><span>.</span></div>';

        div.appendChild(avatarDiv);
        div.appendChild(contentDiv);
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }

    function removeLoadingMsg(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }
});
