// ============================================
// GOPT Web Interface - Frontend Logic
// ============================================

// Global state
let currentFile = null;
let currentTaskId = null;
let statusPollingInterval = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const selectFileBtn = document.getElementById('selectFileBtn');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFileBtn');
const transcriptInput = document.getElementById('transcriptInput');
const submitBtn = document.getElementById('submitBtn');
const transcribeBtn = document.getElementById('transcribeBtn');
const clearTranscriptBtn = document.getElementById('clearTranscriptBtn');
const asrStatus = document.getElementById('asrStatus');
const asrStatusText = document.getElementById('asrStatusText');

const processingSection = document.getElementById('processingSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

const resultsSection = document.getElementById('resultsSection');
const scoresGrid = document.getElementById('scoresGrid');
const phoneScoresTable = document.getElementById('phoneScoresTable');
const rawJsonData = document.getElementById('rawJsonData');
const resultFileName = document.getElementById('resultFileName');
const resultTranscript = document.getElementById('resultTranscript');
const downloadBtn = document.getElementById('downloadBtn');
const newAssessmentBtn = document.getElementById('newAssessmentBtn');

const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const errorDetails = document.getElementById('errorDetails');
const retryBtn = document.getElementById('retryBtn');

const healthCheckLink = document.getElementById('healthCheckLink');
const healthModal = document.getElementById('healthModal');
const closeHealthModal = document.getElementById('closeHealthModal');
const healthModalBody = document.getElementById('healthModalBody');

// ===== Utility Functions =====

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showSection(section) {
    [processingSection, resultsSection, errorSection].forEach(s => {
        s.style.display = 'none';
    });
    if (section) {
        section.style.display = 'block';
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function showError(message, details = '') {
    errorMessage.textContent = message;
    errorDetails.textContent = details;
    showSection(errorSection);
}

function showNotification(message, type = 'info') {
    // Simple console log for now, can be enhanced with toast notifications
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// ===== File Handling =====

function handleFile(file) {
    if (!file) return;

    // Supported audio formats
    const supportedFormats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    
    // Validate file type
    if (!supportedFormats.includes(fileExt)) {
        alert('请上传支持的音频格式：WAV, MP3, M4A, FLAC, OGG, AAC, WMA');
        return;
    }

    // Validate file size (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('文件大小超过 50MB 限制');
        return;
    }

    currentFile = file;
    
    // Update UI
    dropZone.style.display = 'none';
    fileInfo.style.display = 'block';
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    // Show conversion notice for non-WAV files
    if (fileExt !== '.wav') {
        const conversionNotice = document.createElement('p');
        conversionNotice.className = 'conversion-notice';
        conversionNotice.textContent = `注意：${fileExt.toUpperCase()} 文件将自动转换为 16kHz 单声道 WAV 格式`;
        conversionNotice.style.color = '#f57c00';
        conversionNotice.style.fontSize = '0.9rem';
        conversionNotice.style.marginTop = '10px';
        
        // Remove any existing notice
        const existingNotice = document.querySelector('.conversion-notice');
        if (existingNotice) {
            existingNotice.remove();
        }
        
        fileInfo.appendChild(conversionNotice);
    }
    
    // Enable submit button if transcript is provided
    checkFormValidity();
}

function removeFile() {
    currentFile = null;
    fileInput.value = '';
    dropZone.style.display = 'block';
    fileInfo.style.display = 'none';
    checkFormValidity();
}

function checkFormValidity() {
    // Enable submit if file is selected and transcript is not empty
    const hasFile = currentFile !== null;
    const hasTranscript = transcriptInput.value.trim().length > 0;
    submitBtn.disabled = !(hasFile && hasTranscript);
    
    // Enable transcribe button if file is selected
    transcribeBtn.disabled = !hasFile;
    
    // Show/hide clear button
    clearTranscriptBtn.style.display = hasTranscript ? 'inline-block' : 'none';
}

// ===== Event Listeners - File Upload =====

selectFileBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

removeFileBtn.addEventListener('click', removeFile);

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// Auto-convert transcript to uppercase
// Auto-transcribe button
transcribeBtn.addEventListener('click', async () => {
    await transcribeAudio();
});

// Clear transcript button
clearTranscriptBtn.addEventListener('click', () => {
    transcriptInput.value = '';
    checkFormValidity();
});

transcriptInput.addEventListener('input', (e) => {
    const cursorPos = e.target.selectionStart;
    const textBefore = e.target.value.substring(0, cursorPos);
    const textAfter = e.target.value.substring(cursorPos);
    const upperBefore = textBefore.toUpperCase();
    const upperAfter = textAfter.toUpperCase();
    
    e.target.value = upperBefore + upperAfter;
    e.target.setSelectionRange(upperBefore.length, upperBefore.length);
    
    checkFormValidity();
});

// ===== Audio Transcription =====

async function transcribeAudio() {
    if (!currentFile) {
        alert('请先选择音频文件');
        return;
    }

    // Show status
    asrStatus.style.display = 'block';
    asrStatusText.textContent = '正在识别语音，请稍候...';
    transcribeBtn.disabled = true;

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);

        // Call transcribe API
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '识别失败');
        }

        const result = await response.json();
        
        // Fill in the transcript
        transcriptInput.value = result.text;
        
        // Update status
        asrStatusText.textContent = `识别成功！用时 ${result.processing_time.toFixed(1)}秒`;
        asrStatusText.parentElement.style.backgroundColor = '#e8f5e9';
        asrStatusText.parentElement.style.color = '#2e7d32';
        
        // Hide status after 3 seconds
        setTimeout(() => {
            asrStatus.style.display = 'none';
            asrStatusText.parentElement.style.backgroundColor = '#e3f2fd';
            asrStatusText.parentElement.style.color = '#1976d2';
        }, 3000);

        // Check form validity
        checkFormValidity();
        
        showNotification(`语音识别成功: ${result.text}`, 'success');

    } catch (error) {
        console.error('Transcription error:', error);
        asrStatusText.textContent = `识别失败: ${error.message}`;
        asrStatusText.parentElement.style.backgroundColor = '#ffebee';
        asrStatusText.parentElement.style.color = '#c62828';
        
        // Hide status after 5 seconds
        setTimeout(() => {
            asrStatus.style.display = 'none';
            asrStatusText.parentElement.style.backgroundColor = '#e3f2fd';
            asrStatusText.parentElement.style.color = '#1976d2';
        }, 5000);
        
        alert('语音识别失败: ' + error.message);
    } finally {
        transcribeBtn.disabled = false;
    }
}

// ===== Form Submission =====

submitBtn.addEventListener('click', async () => {
    if (!currentFile) {
        alert('请先选择音频文件');
        return;
    }
    
    if (!transcriptInput.value.trim()) {
        alert('请输入音频文本转录，或点击"自动识别"按钮');
        return;
    }

    // Prepare form data
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('transcript', transcriptInput.value.trim());

    try {
        // Show processing section
        showSection(processingSection);
        updateProgress(0, '正在上传文件...');

        // Upload file
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '上传失败');
        }

        const result = await response.json();
        currentTaskId = result.task_id;

        showNotification(`文件上传成功，任务ID: ${currentTaskId}`, 'success');
        
        // Start polling status
        startStatusPolling();

    } catch (error) {
        console.error('Upload error:', error);
        showError('文件上传失败', error.message);
    }
});

// ===== Status Polling =====

function startStatusPolling() {
    if (statusPollingInterval) {
        clearInterval(statusPollingInterval);
    }

    updateProgress(10, '初始化处理流程...');

    statusPollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${currentTaskId}`);
            
            if (!response.ok) {
                throw new Error('无法获取状态');
            }

            const status = await response.json();
            handleStatusUpdate(status);

        } catch (error) {
            console.error('Status polling error:', error);
            clearInterval(statusPollingInterval);
            showError('状态查询失败', error.message);
        }
    }, 2000); // Poll every 2 seconds
}

function handleStatusUpdate(status) {
    console.log('Status update:', status);

    // Check for segmentation info and display progress
    const segProgress = document.getElementById('segmentProgress');
    const segProgressText = document.getElementById('segmentProgressText');
    const processingNote = document.getElementById('processingNote');
    
    if (status.segmentation) {
        const seg = status.segmentation;
        segProgress.style.display = 'block';
        
        if (seg.current_segment) {
            segProgressText.textContent = `正在处理第 ${seg.current_segment}/${seg.num_segments} 段 (总计 ${seg.total_words} 词, 约 ${seg.estimated_phonemes} 音素)`;
            processingNote.textContent = `处理过程可能需要 ${seg.num_segments * 7} 分钟（${seg.num_segments} 段），请耐心等待...`;
        } else {
            segProgressText.textContent = `检测到长音频（约 ${seg.estimated_phonemes} 音素），将分成 ${seg.num_segments} 段处理`;
            processingNote.textContent = `处理过程可能需要 ${seg.num_segments * 7} 分钟（${seg.num_segments} 段），请耐心等待...`;
        }
    } else {
        segProgress.style.display = 'none';
    }

    switch (status.status) {
        case 'uploaded':
            // Check if conversion is being applied
            if (status.stage && status.stage.includes('Converting')) {
                updateProgress(15, '正在转换音频格式...');
            } else if (status.stage && status.stage.includes('Analyzing')) {
                updateProgress(12, '正在分析音频属性...');
            } else {
                updateProgress(20, '文件已上传，等待处理...');
            }
            break;
        case 'ready':
            updateProgress(25, '音频已准备就绪，开始评估...');
            break;
        case 'processing':
            // Parse the stage for more specific progress
            if (status.stage && status.stage.includes('Converting')) {
                updateProgress(15, '正在转换音频格式...');
            } else if (status.stage && status.stage.includes('converted')) {
                updateProgress(30, '音频转换完成，开始提取特征...');
            } else {
                updateProgress(50, status.stage || '正在处理...');
            }
            break;
        case 'completed':
            clearInterval(statusPollingInterval);
            updateProgress(100, '处理完成！');
            setTimeout(() => loadResults(), 500);
            break;
        case 'error':
            clearInterval(statusPollingInterval);
            showError('处理失败', status.error || '未知错误');
            break;
        default:
            updateProgress(30, status.stage || '处理中...');
    }
}

function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
}

// ===== Load and Display Results =====

async function loadResults() {
    try {
        const response = await fetch(`/result/${currentTaskId}`);
        
        if (!response.ok) {
            throw new Error('无法获取结果');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Error loading results:', error);
        showError('结果加载失败', error.message);
    }
}

function displayResults(data) {
    const { filename, transcript, results } = data;

    // Display file info
    resultFileName.textContent = filename;
    resultTranscript.textContent = transcript;

    // Display segmentation info if available
    if (results.segmentation_info && results.segmentation_info.num_segments > 1) {
        const segInfo = results.segmentation_info;
        const segDiv = document.getElementById('segmentationInfo');
        const segDetails = document.getElementById('segmentationDetails');
        
        let details = `共处理 ${segInfo.num_segments} 个段落`;
        if (segInfo.segments) {
            details += `（`;
            segInfo.segments.forEach((seg, i) => {
                details += `段${i+1}: ${seg.word_count}词`;
                if (i < segInfo.segments.length - 1) details += ', ';
            });
            details += `）- 结果已自动整合`;
        }
        
        segDetails.textContent = details;
        segDiv.style.display = 'block';
    } else {
        document.getElementById('segmentationInfo').style.display = 'none';
    }

    // Display utterance scores
    displayUtteranceScores(results.utterance_scores);

    // Display feedback
    displayFeedback(results);

    // Display word scores (handle both dict and array formats)
    if (results.word_scores) {
        const hasWordScores = Array.isArray(results.word_scores) 
            ? results.word_scores.length > 0 
            : (results.word_scores.total && results.word_scores.total.length > 0);
        if (hasWordScores) {
            displayWordScores(results.word_scores);
        }
    }

    // Display phone scores
    displayPhoneScores(results.phone_scores);

    // Display raw JSON
    rawJsonData.textContent = JSON.stringify(results, null, 2);

    // Show results section
    showSection(resultsSection);
}

function displayUtteranceScores(scores) {
    scoresGrid.innerHTML = '';

    const scoreLabels = {
        accuracy: '准确度 Accuracy',
        completeness: '完整度 Completeness',
        fluency: '流畅度 Fluency',
        prosodic: '韵律 Prosodic',
        total: '总分 Total'
    };

    for (const [key, value] of Object.entries(scores)) {
        const scoreValue = typeof value === 'number' ? value : value.predicted || value;
        const normalizedScore = scoreValue; // 0-2 scale
        
        const card = document.createElement('div');
        card.className = `score-card ${key}`;
        card.innerHTML = `
            <div class="score-label">${scoreLabels[key] || key}</div>
            <div class="score-value">${normalizedScore.toFixed(2)}</div>
            <div class="score-scale">0-2 scale</div>
        `;
        scoresGrid.appendChild(card);
    }
}

function displayWordScores(wordScores) {
    const wordScoresSection = document.getElementById('wordScoresSection');
    const wordSentence = document.getElementById('wordSentence');
    
    wordSentence.innerHTML = '';

    // Handle both dict-of-arrays format and list-of-objects format
    let wordDataList = [];
    
    if (Array.isArray(wordScores)) {
        // Legacy format: list of objects with word_text, total, etc.
        wordDataList = wordScores;
    } else if (wordScores && typeof wordScores === 'object') {
        // Pipeline format: dict of arrays {accuracy: [...], stress: [...], total: [...]}
        const totals = wordScores.total || [];
        const accuracies = wordScores.accuracy || [];
        const stresses = wordScores.stress || [];
        
        const count = totals.length;
        for (let i = 0; i < count; i++) {
            wordDataList.push({
                word_text: `Phone ${i+1}`,
                accuracy: accuracies[i] || 0,
                stress: stresses[i] || 0,
                total: totals[i] || 0,
                phone_count: 1
            });
        }
    }
    
    if (wordDataList.length === 0) {
        wordScoresSection.style.display = 'none';
        return;
    }

    wordDataList.forEach((wordData, index) => {
        const wordSpan = document.createElement('span');
        wordSpan.className = 'word-item';
        
        // Determine color class based on total score
        let colorClass = 'score-medium';
        if (wordData.total >= 1.5) colorClass = 'score-high';
        else if (wordData.total >= 1.0) colorClass = 'score-good';
        else if (wordData.total >= 0.5) colorClass = 'score-medium';
        else if (wordData.total >= 0.0) colorClass = 'score-low';
        else colorClass = 'score-very-low';
        
        wordSpan.classList.add(colorClass);
        wordSpan.textContent = wordData.word_text || `Word ${index+1}`;
        wordSpan.title = `总分: ${(wordData.total || 0).toFixed(2)}`;
        
        // Store word data for tooltip
        wordSpan.dataset.wordData = JSON.stringify(wordData);
        
        // Click event to show tooltip
        wordSpan.addEventListener('click', function(e) {
            showWordTooltip(wordData, e);
        });
        
        wordSentence.appendChild(wordSpan);
        
        // Add space between words (except after last word)
        if (index < wordDataList.length - 1) {
            const space = document.createTextNode(' ');
            wordSentence.appendChild(space);
        }
    });

    // Show word scores section
    wordScoresSection.style.display = 'block';
}

function showWordTooltip(wordData, event) {
    const tooltip = document.getElementById('wordTooltip');
    const tooltipWord = tooltip.querySelector('.tooltip-word');
    
    // Update tooltip content
    tooltipWord.textContent = wordData.word_text;
    document.getElementById('tooltipAccuracy').textContent = wordData.accuracy.toFixed(3);
    document.getElementById('tooltipStress').textContent = wordData.stress.toFixed(3);
    document.getElementById('tooltipTotal').textContent = wordData.total.toFixed(3);
    document.getElementById('tooltipPhoneCount').textContent = wordData.phone_count;
    
    // Position tooltip near the clicked word
    const rect = event.target.getBoundingClientRect();
    tooltip.style.display = 'block';
    tooltip.style.position = 'fixed';
    tooltip.style.left = `${rect.left}px`;
    tooltip.style.top = `${rect.bottom + 10}px`;
    
    // Adjust if tooltip goes off screen
    setTimeout(() => {
        const tooltipRect = tooltip.getBoundingClientRect();
        if (tooltipRect.right > window.innerWidth) {
            tooltip.style.left = `${window.innerWidth - tooltipRect.width - 20}px`;
        }
        if (tooltipRect.bottom > window.innerHeight) {
            tooltip.style.top = `${rect.top - tooltipRect.height - 10}px`;
        }
    }, 0);
}

function hideWordTooltip() {
    const tooltip = document.getElementById('wordTooltip');
    tooltip.style.display = 'none';
}

// Close tooltip when clicking outside
document.addEventListener('click', function(e) {
    const tooltip = document.getElementById('wordTooltip');
    if (tooltip && !e.target.closest('.word-item') && !e.target.closest('.word-tooltip')) {
        hideWordTooltip();
    }
});

function displayPhoneScores(phoneScores) {
    phoneScoresTable.innerHTML = '';

    // Display first 20 phone scores
    const displayScores = phoneScores.slice(0, 20);

    displayScores.forEach((score, index) => {
        // Skip invalid scores
        if (score === 0 || score === -1) return;

        const item = document.createElement('div');
        
        // Determine color class based on score
        let colorClass = 'medium';
        if (score >= 1.8) colorClass = 'high';
        else if (score < 1.0) colorClass = 'low';
        
        item.className = `phone-score-item ${colorClass}`;
        item.innerHTML = `
            <div class="phone-index">Phone ${index + 1}</div>
            <div class="phone-value">${score.toFixed(2)}</div>
        `;
        phoneScoresTable.appendChild(item);
    });

    // If no valid scores found
    if (phoneScoresTable.children.length === 0) {
        phoneScoresTable.innerHTML = '<p style="text-align: center; color: #999;">暂无音素级评分数据</p>';
    }
}

// ===== Feedback Generation =====

function displayFeedback(results) {
    const feedbackSection = document.getElementById('feedbackSection');
    const feedbackCard = document.getElementById('feedbackCard');
    
    if (!feedbackSection || !feedbackCard) return;
    
    // Calculate feedback level
    const feedback = calculateFeedback(results);
    
    // Clear previous content
    feedbackCard.className = 'feedback-card ' + feedback.level;
    feedbackCard.innerHTML = `
        <span class="feedback-icon">${feedback.icon}</span>
        <span class="feedback-level ${feedback.level}">${feedback.levelLabel}</span>
        <div class="feedback-title">${feedback.title}</div>
        <div class="feedback-description">${feedback.description}</div>
        ${feedback.highlights ? `
            <div class="feedback-highlights ${feedback.level}">
                <h4>${feedback.highlightsTitle}</h4>
                <ul>
                    ${feedback.highlights.map(h => `<li>${h}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
        ${feedback.suggestions && feedback.suggestions.length > 0 ? `
            <div class="feedback-suggestions">
                <h4>我的建议是：</h4>
                <ul>
                    ${feedback.suggestions.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
    `;
    
    // Show feedback section
    feedbackSection.style.display = 'block';
}

function calculateFeedback(results) {
    const utteranceScores = results.utterance_scores || {};
    const wordScores = results.word_scores || [];
    const phoneScores = results.phone_scores || [];
    
    // Get scores (handle both number and object formats)
    const totalScore = typeof utteranceScores.total === 'number' 
        ? utteranceScores.total 
        : (utteranceScores.total?.predicted || utteranceScores.total || 0);
    const accuracy = typeof utteranceScores.accuracy === 'number' 
        ? utteranceScores.accuracy 
        : (utteranceScores.accuracy?.predicted || utteranceScores.accuracy || 0);
    const completeness = typeof utteranceScores.completeness === 'number' 
        ? utteranceScores.completeness 
        : (utteranceScores.completeness?.predicted || utteranceScores.completeness || 0);
    const fluency = typeof utteranceScores.fluency === 'number' 
        ? utteranceScores.fluency 
        : (utteranceScores.fluency?.predicted || utteranceScores.fluency || 0);
    const prosodic = typeof utteranceScores.prosodic === 'number' 
        ? utteranceScores.prosodic 
        : (utteranceScores.prosodic?.predicted || utteranceScores.prosodic || 0);
    
    // Calculate average scores
    const avgScore = (accuracy + completeness + fluency + prosodic) / 4;
    
    // Analyze word scores (handle both array and dict formats)
    let validWordScores = [];
    if (Array.isArray(wordScores)) {
        validWordScores = wordScores.filter(w => w.total && w.total > 0);
    } else if (wordScores && wordScores.total && Array.isArray(wordScores.total)) {
        validWordScores = wordScores.total.filter(t => t > 0).map(t => ({ total: t }));
    }
    const avgWordScore = validWordScores.length > 0
        ? validWordScores.reduce((sum, w) => sum + w.total, 0) / validWordScores.length
        : 0;
    
    // Analyze phone scores
    const validPhoneScores = phoneScores.filter(p => p > 0 && p !== -1);
    const avgPhoneScore = validPhoneScores.length > 0
        ? validPhoneScores.reduce((sum, p) => sum + p, 0) / validPhoneScores.length
        : 0;
    
    // Determine feedback level based on total score and consistency
    // Level thresholds: Excellent (>= 1.5), Good (>= 0.8), Needs Improvement (< 0.8)
    // Handle negative scores by treating them as low scores
    const normalizedTotalScore = Math.max(0, totalScore);
    const normalizedAvgScore = Math.max(0, avgScore);
    const normalizedAvgWordScore = Math.max(0, avgWordScore);
    
    let level, levelLabel, icon, title, description, highlightsTitle, highlights, suggestions;
    
    if (normalizedTotalScore >= 1.5 && normalizedAvgScore >= 1.4 && (normalizedAvgWordScore >= 1.3 || normalizedAvgWordScore === 0)) {
        // Excellent level
        level = 'excellent';
        levelLabel = '优秀';
        icon = '🌟';
        title = '太棒了！你的发音非常出色！';
        description = `听完你的录音，我真的很惊喜！总分 ${normalizedTotalScore.toFixed(2)} 分，这已经是一个非常优秀的成绩了。你的发音清晰自然，听起来就像母语者一样流畅。每个单词都发音到位，语音的完整性也很好，最重要的是语速和节奏把握得非常好。`;
        highlightsTitle = '让我来具体说说你做得好的地方：';
        highlights = [
            `你的准确度达到了 ${accuracy.toFixed(2)}，说明音素发音很标准，几乎没有错误`,
            `完整度 ${completeness.toFixed(2)} 特别高，这意味着你把每个单词都说完整了`,
            `流畅度 ${fluency.toFixed(2)} 说明你的语速控制得很好，听起来很自然`,
            `韵律 ${prosodic.toFixed(2)} 很棒，重音和语调都把握得很准确`,
            `总体来说，你的英语发音已经达到了很高的水平`
        ];
        suggestions = [
            '继续保持这个发音标准，你已经在正确的路上了',
            '可以挑战一些更长、更复杂的句子，锻炼连续语音的能力',
            '多和母语者交流，进一步感受自然对话的节奏'
        ];
    } else if (normalizedTotalScore >= 0.8 && normalizedAvgScore >= 0.7) {
        // Good level
        level = 'good';
        levelLabel = '良好';
        icon = '👍';
        title = '很不错！你的发音基础很扎实';
        description = `你的总分是 ${normalizedTotalScore.toFixed(2)} 分，这已经是一个不错的成绩了！我能听懂你说的大部分内容，发音也比较清晰。看得出来你在发音上花了不少功夫，整体水平已经很接近优秀了。不过还有一些细节可以再打磨一下，相信通过练习会越来越好。`;
        highlightsTitle = '你做得不错的地方：';
        highlights = [
            `整体发音清晰，我能够清楚地理解你的意思`,
            `大部分单词都发音正确，基础很好`,
            `完整度保持得不错，单词都说完整了`,
            `语速适中，不会太快也不会太慢`
        ];
        
        // Generate specific suggestions based on lower scores
        suggestions = [];
        if (accuracy < 1.0) {
            suggestions.push(`准确度目前是 ${accuracy.toFixed(2)}，可以更仔细地练习每个音素的发音，特别是那些容易混淆的音`);
        }
        if (fluency < 0.8) {
            suggestions.push(`流畅度 ${fluency.toFixed(2)} 还可以提升，试着让语速更自然一些，减少不必要的停顿`);
        }
        if (prosodic < 0.8) {
            suggestions.push(`韵律方面（${prosodic.toFixed(2)}）需要加强，多注意单词的重音位置和句子的语调起伏`);
        }
        if (completeness < 0.8) {
            suggestions.push(`完整度可以再提高一些（当前 ${completeness.toFixed(2)}），确保所有单词都清晰发音`);
        }
        if (suggestions.length === 0) {
            suggestions.push('继续练习以提升整体发音水平');
            suggestions.push('建议你多听一些标准的英语发音，然后模仿他们的语音语调');
        }
    } else {
        // Needs Improvement level
        level = 'needs-improvement';
        levelLabel = '需改进';
        icon = '💪';
        title = '加油！我们一起努力改进发音';
        description = `我看到你的总分是 ${normalizedTotalScore.toFixed(2)} 分，虽然还有提升空间，但请别灰心！每个人的学习之路都不一样，重要的是你现在开始关注发音了。我注意到有些地方需要特别注意，不过没关系，通过系统的练习，你一定能够取得明显进步的。相信我，坚持下去就会有收获！`;
        highlightsTitle = '我们需要一起关注这些问题：';
        highlights = [];
        
        // Add specific issues
        if (accuracy < 0.5) {
            highlights.push(`准确度（${accuracy.toFixed(2)}）需要重点提升，有些音素的发音需要纠正`);
        }
        if (completeness < 0.5) {
            highlights.push(`完整度（${completeness.toFixed(2)}）还可以提高，部分单词可能未清晰发音`);
        }
        if (fluency < 0.5) {
            highlights.push(`流畅度（${fluency.toFixed(2)}）还不够，语速控制和平稳度都需要练习`);
        }
        if (prosodic < 0.5) {
            highlights.push(`韵律（${prosodic.toFixed(2)}）方面要加强，重音和语调的变化要更自然`);
        }
        if (avgWordScore > 0 && avgWordScore < 0.8) {
            highlights.push(`单词发音需要改进（平均分 ${avgWordScore.toFixed(2)}），建议逐个单词练习`);
        }
        if (highlights.length === 0) {
            highlights.push('整体来说，基础发音技能还需要进一步巩固');
        }
        
        suggestions = [
            '每天坚持练习 15-20 分钟，循序渐进，不要急于求成',
            '重点练习那些得分较低的单词，一个一个攻克',
            '多听标准英语发音，可以看英语视频、听英语播客，感受自然的语音',
            '尝试录音练习，然后对比标准发音，找出差异',
            '可以找一些发音教程或者使用专门的发音学习应用，系统性地学习',
            '记住，发音是一个过程，每天进步一点点就够了'
        ];
    }
    
    return {
        level,
        levelLabel,
        icon,
        title,
        description,
        highlightsTitle,
        highlights,
        suggestions
    };
}

// ===== Download Results =====

downloadBtn.addEventListener('click', () => {
    if (!currentTaskId) return;
    
    window.location.href = `/download/${currentTaskId}`;
});

// ===== New Assessment =====

function resetForm() {
    currentFile = null;
    currentTaskId = null;
    
    if (statusPollingInterval) {
        clearInterval(statusPollingInterval);
        statusPollingInterval = null;
    }
    
    fileInput.value = '';
    transcriptInput.value = '';
    dropZone.style.display = 'block';
    fileInfo.style.display = 'none';
    submitBtn.disabled = true;
    
    showSection(null);
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

newAssessmentBtn.addEventListener('click', resetForm);
retryBtn.addEventListener('click', resetForm);

// ===== Health Check =====

healthCheckLink.addEventListener('click', async (e) => {
    e.preventDefault();
    
    healthModal.style.display = 'flex';
    healthModalBody.innerHTML = '<p>正在检查系统状态...</p>';
    
    try {
        const response = await fetch('/health');
        const health = await response.json();
        
        const dockerStatus = health.docker_running 
            ? '<span style="color: green;">✓ 运行中</span>' 
            : '<span style="color: red;">✗ 未运行</span>';
        
        healthModalBody.innerHTML = `
            <div style="line-height: 2;">
                <p><strong>系统状态:</strong> ${health.status}</p>
                <p><strong>Docker 容器:</strong> ${health.docker_container}</p>
                <p><strong>Docker 状态:</strong> ${dockerStatus}</p>
                <p><strong>检查时间:</strong> ${new Date(health.timestamp).toLocaleString('zh-CN')}</p>
            </div>
            ${!health.docker_running ? `
                <div style="margin-top: 20px; padding: 15px; background: #fff3e0; border-radius: 8px;">
                    <p style="color: #f39c12; margin: 0;">
                        ⚠️ Docker 容器未运行，请先启动容器：<br>
                        <code style="display: block; margin-top: 10px; padding: 10px; background: #fff; border-radius: 4px;">
                            docker start ${health.docker_container}
                        </code>
                    </p>
                </div>
            ` : ''}
        `;
        
    } catch (error) {
        healthModalBody.innerHTML = `
            <p style="color: red;">❌ 无法连接到服务器</p>
            <p style="color: #666;">${error.message}</p>
        `;
    }
});

closeHealthModal.addEventListener('click', () => {
    healthModal.style.display = 'none';
});

// Close modal when clicking outside
healthModal.addEventListener('click', (e) => {
    if (e.target === healthModal) {
        healthModal.style.display = 'none';
    }
});

// ===== Initialization =====

document.addEventListener('DOMContentLoaded', () => {
    console.log('GOPT Web Interface loaded');
    
    // Check server health on load
    fetch('/health')
        .then(response => response.json())
        .then(health => {
            if (!health.docker_running) {
                console.warn('Docker container is not running');
            }
        })
        .catch(error => {
            console.error('Failed to check server health:', error);
        });
});

// ===== Cleanup on page unload =====

window.addEventListener('beforeunload', () => {
    if (statusPollingInterval) {
        clearInterval(statusPollingInterval);
    }
});

