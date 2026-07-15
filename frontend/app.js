const API_URL = window.location.origin;
const AVATARS = {
    user: '用户.png',
    assistant: 'AI.png'
};
const STATUS = {
    idle: {
        color: '#3fa46a',
        text: '在线'
    },
    loading: {
        color: '#ef8f52',
        text: '思考中'
    }
};
const PROFILE_HIGHLIGHT_DURATION = 30000;
const TOAST_DURATION = 30000;
const PROFILE_LABELS = {};
const ROOT_PROFILE_ORDER = ['state_axis', 'context_axis'];
const HIDDEN_PROFILE_KEYS = new Set(['persona_type', 'persona_name']);
const STATIC_PROFILE_LAYER_ORDER = ['core', 'regulation', 'cognition', 'identity', 'behavior'];
const DEFAULT_EXPANDED_PATHS = [
    'state_axis',
    'state_axis.static_profile',
    'state_axis.current_state',
    'state_axis.static_profile.core',
    'state_axis.static_profile.identity',
];
const DEFAULT_SYSTEM_MESSAGE = '对话已准备好。你好！我是你的个性化助手，请尽情聊天吧！';

const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistory = document.getElementById('chatHistory');
const profileContent = document.getElementById('profileContent');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const experimentPanel = document.getElementById('experimentPanel');
const experimentResetBtn = document.getElementById('experimentResetBtn');
const clearAblationBtn = document.getElementById('clearAblationBtn');
const ablationStatusText = document.getElementById('ablationStatusText');
const ablationInputs = Array.from(document.querySelectorAll('input[name="ablateDimension"]'));
const characterCards = document.getElementById('characterCards');

let isLoading = false;
let selectedProfileId = null;
let selectedPersonaId = null;
let hasInitializedProfileTree = false;
let profileHighlightTimer = null;
let ablationSelectionUnlocked = false;
let ablationSelectionLocked = false;
const expandedProfilePaths = new Set(DEFAULT_EXPANDED_PATHS);

document.addEventListener('DOMContentLoaded', () => {
    document.body.classList.add('character-picking');
    setChatEnabled(false);
    initializeCharacterCards();
    updateStatus('idle');
    syncExperimentToggleIcon();

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !isLoading) {
            sendMessage();
        }
    });

    if (experimentResetBtn) {
        experimentResetBtn.addEventListener('click', resetExperimentBaseline);
    }
    if (clearAblationBtn) {
        clearAblationBtn.addEventListener('click', clearAblationSelection);
    }
    if (experimentPanel) {
        experimentPanel.addEventListener('toggle', syncExperimentToggleIcon);
    }
    ablationInputs.forEach((input) => {
        input.addEventListener('change', updateAblationStatus);
    });
    window.addEventListener('pagehide', finalizeSessionOnUnload);
    window.addEventListener('beforeunload', finalizeSessionOnUnload);
    syncAblationControls();
});

function finalizeSessionOnUnload() {
    const url = `${API_URL}/api/finalize-session`;
    if (navigator.sendBeacon) {
        navigator.sendBeacon(url, new Blob(['{}'], { type: 'application/json' }));
        return;
    }
    fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
        keepalive: true
    }).catch(() => {});
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isLoading) { voiceMode = false; return; }
    if (!selectedProfileId) {
        showUpdateNotification([], 'Please select a character first.');
        return;
    }
    if (!voiceMode) systemStartTime = Date.now();

    addMessageToChat(message, 'user');
    userInput.value = '';
    await runAIResponse(message);
}

async function runAIResponse(message) {
    if (!selectedProfileId) {
        showUpdateNotification([], 'Please select a character first.');
        voiceMode = false;
        return;
    }
    const ablateDimension = getSelectedAblationDimension();

    isLoading = true;
    sendBtn.disabled = true;
    userInput.disabled = true;
    updateStatus('loading');

    try {
        const assistantContent = addMessageToChat('', 'assistant');

        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-System-Start': String(systemStartTime || Date.now()),
            },
            body: JSON.stringify({
                message,
                character_id: selectedProfileId, persona_id: selectedPersonaId,
                ablate_dimension: ablateDimension
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '请求失败');
        }

        const onToken = voiceMode ? feedTTSToken : null;
        const data = await readChatStream(response, assistantContent, onToken);
        if (voiceMode) { finalizeTTS(); voiceMode = false; }
        const highlightFields = Array.isArray(data.updated_fields) ? data.updated_fields : [];
        await loadProfile(highlightFields);
        if (highlightFields.length > 0) {
            showUpdateNotification(highlightFields);
        }
    } catch (error) {
        console.error('错误:', error);
        addMessageToChat(`错误: ${error.message}`, 'system');
    } finally {
        if (ablateDimension) {
            ablationSelectionLocked = true;
            syncAblationControls();
        }
        isLoading = false;
        sendBtn.disabled = false;
        userInput.disabled = false;
        updateStatus('idle');
        userInput.focus();
    }
}

let availablePersonas = [];

async function initializeCharacterCards() {
    try {
        const response = await fetch(`${API_URL}/api/characters`);
        if (!response.ok) throw new Error('failed to load characters');

        const data = await response.json();
        availablePersonas = data.agent_personas || [];
        renderCharacterCards(data.user_profiles || [], data.active_profile_id || null);
        renderPersonaSelector(availablePersonas, data.active_persona_id || null);
    } catch (error) {
        console.error('load characters failed:', error);
        if (characterCards) {
            characterCards.innerHTML = '<div class="loading" style="color:#d85d59;">Failed to load characters</div>';
        }
    }
}

function renderPersonaSelector(personas, activePersonaId) {
    let selector = document.getElementById('personaSelector');
    if (!selector) {
        selector = document.createElement('div');
        selector.id = 'personaSelector';
        selector.className = 'persona-selector';
        const characterStrip = document.querySelector('.character-strip');
        if (characterStrip) {
            characterStrip.appendChild(selector);
        }
    }
    selector.innerHTML = '';

    const label = document.createElement('span');
    label.className = 'persona-selector-label';
    label.textContent = 'Agent Persona';
    selector.appendChild(label);

    personas.forEach((persona) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'persona-pill';
        btn.dataset.personaId = persona.id;
        btn.textContent = persona.display_name;
        if (persona.id === activePersonaId) {
            btn.classList.add('active');
            selectedPersonaId = persona.id;
        }
        btn.addEventListener('click', () => selectPersona(persona.id));
        selector.appendChild(btn);
    });

    if (!selectedPersonaId && personas.length > 0) {
        selectedPersonaId = personas[0].id;
        selector.querySelector('.persona-pill')?.classList.add('active');
    }
}

function selectPersona(personaId) {
    selectedPersonaId = personaId;
    document.querySelectorAll('.persona-pill').forEach((pill) => {
        pill.classList.toggle('active', pill.dataset.personaId === personaId);
    });
    // If a profile is already selected, rebuild agent with new persona
    if (selectedProfileId) {
        selectCharacter(selectedProfileId);
    }
}

function renderCharacterCards(characters, activeCharacterId) {
    if (!characterCards) return;
    characterCards.innerHTML = '';

    if (!characters.length) {
        characterCards.innerHTML = '<div class="loading">No characters available</div>';
        return;
    }

    characters.forEach((character) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'character-card';
        card.dataset.characterId = character.id;

        const initial = document.createElement('span');
        initial.className = 'character-avatar';
        initial.textContent = (character.display_name || character.name || '?').slice(0, 1).toUpperCase();

        const copy = document.createElement('span');
        copy.className = 'character-copy';

        const name = document.createElement('span');
        name.className = 'character-name';
        name.textContent = character.display_name || character.name;

        const desc = document.createElement('span');
        desc.className = 'character-desc';
        desc.textContent = character.description || '';

        copy.appendChild(name);
        copy.appendChild(desc);
        card.appendChild(initial);
        card.appendChild(copy);
        card.addEventListener('click', () => selectCharacter(character.id));
        characterCards.appendChild(card);
    });

    if (activeCharacterId) {
        selectCharacter(activeCharacterId);
    }
}



async function selectCharacter(profileId) {
    if (!profileId || isLoading) return;

    const personaId = selectedPersonaId || (availablePersonas.length > 0 ? availablePersonas[0].id : null);
    if (!personaId) {
        showUpdateNotification([], 'No agent persona available.');
        return;
    }

    try {
        setChatEnabled(false);
        const response = await fetch(`${API_URL}/api/characters/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile_id: profileId, persona_id: personaId })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'failed to select character');
        }

        selectedProfileId = profileId;
        document.body.classList.remove('character-picking');
        document.querySelectorAll('.character-card').forEach((card) => {
            card.classList.toggle('active', card.dataset.characterId === profileId);
        });

        clearConversationView();
        renderProfile(data.profile || {}, []);
        setChatEnabled(true);
        const profileLabel = data.profile_name || profileId;
        const personaLabel = data.persona_name || personaId;
        showUpdateNotification([], `${profileLabel} x ${personaLabel} ready.`);
    } catch (error) {
        console.error('select character failed:', error);
        showUpdateNotification([], `Character load failed: ${error.message}`);
    }
}

function setChatEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    if (!enabled && !selectedProfileId) {
        userInput.placeholder = 'Select a character card first';
    } else {
        userInput.placeholder = 'Type a message and press Enter';
    }
}

function getSelectedAblationDimension() {
    const selected = ablationInputs.find((input) => input.checked);
    return selected ? selected.value : null;
}

function clearAblationSelection() {
    if (ablationSelectionLocked) return;
    ablationInputs.forEach((input) => {
        input.checked = false;
    });
    updateAblationStatus();
}

async function readChatStream(response, contentNode, onToken) {
    if (!response.body) {
        const data = await response.json();
        contentNode.textContent = data.message || '';
        return data;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalData = null;
    let firstTokenAt = null;

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            const event = JSON.parse(trimmed);
            if (event.type === 'token') {
                if (firstTokenAt === null) {
                    firstTokenAt = performance.now();
                    if (pipeAsrCompleteMs && !pipeFirstTokenMs) {
                        pipeFirstTokenMs = Date.now();
                    }
                }
                contentNode.textContent += event.content || '';
                if (onToken) onToken(event.content || '');
                chatHistory.scrollTop = chatHistory.scrollHeight;
            } else if (event.type === 'done') {
                finalData = event;
                if (event.message) {
                    contentNode.textContent = event.message;
                }
            } else if (event.type === 'error') {
                throw new Error(event.error || 'stream error');
            }
        }
    }

    if (buffer.trim()) {
        const event = JSON.parse(buffer.trim());
        if (event.type === 'done') {
            finalData = event;
        } else if (event.type === 'error') {
            throw new Error(event.error || 'stream error');
        }
    }

    if (!finalData) {
        throw new Error('stream ended without done event');
    }

    return finalData;
}

function addMessageToChat(text, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    if (role === 'system') {
        const systemContent = document.createElement('div');
        systemContent.className = 'message-content';
        systemContent.textContent = text;
        messageDiv.appendChild(systemContent);
    } else {
        const avatar = document.createElement('img');
        avatar.className = 'message-avatar';
        avatar.src = role === 'user' ? AVATARS.user : AVATARS.assistant;
        avatar.alt = role === 'user' ? '用户头像' : 'AI头像';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        const meta = document.createElement('div');
        meta.className = 'message-meta';
        meta.textContent = role === 'user' ? '你' : 'AI 助手';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;

        bubble.appendChild(meta);
        bubble.appendChild(contentDiv);

        if (role === 'user') {
            messageDiv.appendChild(bubble);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(bubble);
        }
    }

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return messageDiv.querySelector('.message-content');
}

async function loadProfile(highlightFields = []) {
    try {
        const response = await fetch(`${API_URL}/api/profile`);
        if (!response.ok) throw new Error('加载画像失败');

        const profile = await response.json();
        renderProfile(profile, highlightFields);
        scheduleHighlightCleanup();
    } catch (error) {
        console.error('加载画像失败:', error);
        profileContent.innerHTML = '<div class="loading" style="color:#d85d59;">加载失败</div>';
    }
}

function renderProfile(profile, highlightFields = []) {
    profileContent.innerHTML = '';

    if (!profile || Object.keys(profile).length === 0) {
        profileContent.innerHTML = '<div class="loading">暂无画像信息</div>';
        return;
    }

    syncExpandedPaths(highlightFields);

    const sections = getOrderedRootEntries(profile)
        .filter(([, value]) => hasDisplayableContent(value))
        .map(([key, value]) => createBranchNode(getLabelForKey(key), value, key, highlightFields, true))
        .filter(Boolean);

    if (sections.length === 0) {
        profileContent.innerHTML = '<div class="loading">暂无画像信息</div>';
        return;
    }

    sections.forEach((section) => profileContent.appendChild(section));
}

function isProfileLeaf(value) {
    return isPlainObject(value) && 'value' in value && 'memory_ids' in value;
}

function createProfileNode(label, value, path, highlightFields = []) {
    if (!hasDisplayableContent(value)) {
        return null;
    }

    if (isProfileLeaf(value)) {
        return createLeafNode(label, value.value, path, highlightFields, value.memory_ids);
    }

    if (Array.isArray(value) && value.every(isPrimitiveValue)) {
        return createTagGroupNode(label, value, path, highlightFields);
    }

    if (Array.isArray(value) || isPlainObject(value)) {
        return createBranchNode(label, value, path, highlightFields, false);
    }

    return createLeafNode(label, value, path, highlightFields, []);
}

function createBranchNode(label, value, path, highlightFields = [], isSection = false) {
    const node = document.createElement('details');
    node.className = isSection ? 'profile-item profile-section' : 'profile-tree-node profile-tree-branch';
    node.dataset.path = path;

    const expanded = expandedProfilePaths.has(path);
    node.open = expanded;

    if (shouldHighlightPath(path, highlightFields)) {
        node.classList.add('highlight');
    }

    node.classList.add(expanded ? 'is-expanded' : 'is-collapsed');

    const summary = document.createElement('summary');
    summary.className = isSection ? 'profile-section-toggle' : 'profile-tree-toggle';

    const heading = document.createElement('div');
    heading.className = isSection ? 'profile-section-heading' : 'profile-tree-heading';

    const arrow = document.createElement('span');
    arrow.className = 'profile-tree-arrow';
    arrow.textContent = expanded ? '▾' : '▸';

    const titleEl = document.createElement('span');
    titleEl.className = isSection ? 'profile-item-title' : 'profile-tree-label';
    titleEl.textContent = label;

    const meta = document.createElement('span');
    meta.className = 'profile-tree-meta';
    meta.textContent = getNodeMeta(value);

    heading.appendChild(arrow);
    heading.appendChild(titleEl);
    summary.appendChild(heading);
    summary.appendChild(meta);

    const children = document.createElement('div');
    children.className = isSection ? 'profile-item-content profile-tree-children' : 'profile-tree-children';

    getNodeEntries(value, path).forEach(({ label: childLabel, value: childValue, path: childPath }) => {
        const childNode = createProfileNode(childLabel, childValue, childPath, highlightFields);
        if (childNode) {
            children.appendChild(childNode);
        }
    });

    node.addEventListener('toggle', () => {
        const nextExpanded = node.open;
        arrow.textContent = nextExpanded ? '▾' : '▸';
        node.classList.toggle('is-expanded', nextExpanded);
        node.classList.toggle('is-collapsed', !nextExpanded);

        if (nextExpanded) {
            expandedProfilePaths.add(path);
        } else {
            expandedProfilePaths.delete(path);
        }
    });

    node.appendChild(summary);
    node.appendChild(children);
    return node;
}

function createLeafNode(label, value, path, highlightFields = [], memoryIds = []) {
    const node = document.createElement('div');
    node.className = 'profile-tree-node profile-tree-leaf';
    node.dataset.path = path;

    if (shouldHighlightPath(path, highlightFields)) {
        node.classList.add('highlight');
    }

    const row = document.createElement('div');
    row.className = 'profile-tree-row';

    const labelEl = document.createElement('span');
    labelEl.className = 'profile-tree-key';
    labelEl.textContent = label;

    const valueEl = document.createElement('div');
    valueEl.className = 'profile-tree-value';

    if (Array.isArray(value)) {
        const tagList = document.createElement('div');
        tagList.className = 'tag-list';
        value.forEach((item) => {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.textContent = formatLeafValue(item);
            tagList.appendChild(tag);
        });
        valueEl.appendChild(tagList);
    } else {
        valueEl.textContent = formatLeafValue(value);
    }

    row.appendChild(labelEl);
    row.appendChild(valueEl);

    if (memoryIds && memoryIds.length > 0) {
        const badge = document.createElement('span');
        badge.className = 'memory-badge';
        badge.title = memoryIds.join(', ');
        badge.textContent = `${memoryIds.length}`;
        row.appendChild(badge);
    }

    node.appendChild(row);
    return node;
}

function createTagGroupNode(label, values, path, highlightFields = []) {
    const node = document.createElement('div');
    node.className = 'profile-tree-node profile-tree-leaf profile-tree-tag-group';
    node.dataset.path = path;

    if (shouldHighlightPath(path, highlightFields)) {
        node.classList.add('highlight');
    }

    const labelEl = document.createElement('div');
    labelEl.className = 'profile-tree-key';
    labelEl.textContent = label;

    const list = document.createElement('div');
    list.className = 'tag-list';

    values.forEach((value) => {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = formatLeafValue(value);
        list.appendChild(tag);
    });

    node.appendChild(labelEl);
    node.appendChild(list);
    return node;
}

function getNodeEntries(value, parentPath) {
    if (Array.isArray(value)) {
        return value.map((item, index) => ({
            label: `[${index}]`,
            value: item,
            path: joinPath(parentPath, String(index))
        }));
    }

    const order = parentPath === 'state_axis.static_profile' ? STATIC_PROFILE_LAYER_ORDER : null;
    const keys = order
        ? [...order.filter((key) => key in value), ...Object.keys(value).filter((key) => !order.includes(key))]
        : Object.keys(value);

    return keys.map((key) => ({
        label: getLabelForKey(key),
        value: value[key],
        path: joinPath(parentPath, key)
    }));
}

function getOrderedRootEntries(profile) {
    const priority = ROOT_PROFILE_ORDER
        .filter((key) => Object.hasOwn(profile, key) && !HIDDEN_PROFILE_KEYS.has(key))
        .map((key) => [key, profile[key]]);
    const remainder = Object.entries(profile)
        .filter(([key]) => !ROOT_PROFILE_ORDER.includes(key) && !HIDDEN_PROFILE_KEYS.has(key));

    return [...priority, ...remainder];
}

function joinPath(parentPath, key) {
    return parentPath ? `${parentPath}.${key}` : key;
}

function getLabelForKey(key) {
    return PROFILE_LABELS[key] || key;
}

function getNodeMeta(value) {
    if (Array.isArray(value)) {
        return `${value.length}项`;
    }

    if (isPlainObject(value)) {
        return `${Object.keys(value).length}项`;
    }

    return '';
}

function hasDisplayableContent(value) {
    if (value === null || value === undefined || value === '') {
        return false;
    }

    if (isProfileLeaf(value)) {
        return hasDisplayableContent(value.value);
    }

    if (Array.isArray(value)) {
        return value.length > 0 && value.some((item) => hasDisplayableContent(item));
    }

    if (isPlainObject(value)) {
        return Object.values(value).some((item) => hasDisplayableContent(item));
    }

    return true;
}

function isPlainObject(value) {
    return Object.prototype.toString.call(value) === '[object Object]';
}

function isPrimitiveValue(value) {
    return value === null || value === undefined || ['string', 'number', 'boolean'].includes(typeof value);
}

function formatLeafValue(value) {
    if (typeof value === 'boolean') {
        return value ? 'true' : 'false';
    }

    return String(value);
}

function normalizePath(path = '') {
    return String(path)
        .replace(/\[(\d+)\]/g, '.$1')
        .replace(/^\./, '')
        .trim();
}

function shouldHighlightPath(path, highlightFields = []) {
    const normalizedPath = normalizePath(path);
    const nodeLabel = normalizedPath.split('.').pop() || normalizedPath;

    return highlightFields
        .map(normalizePath)
        .some((field) =>
            field === normalizedPath ||
            field.startsWith(`${normalizedPath}.`) ||
            normalizedPath.startsWith(`${field}.`) ||
            field.endsWith(`.${nodeLabel}`) ||
            field === nodeLabel
        );
}

function syncExpandedPaths(highlightFields = []) {
    if (!hasInitializedProfileTree) {
        DEFAULT_EXPANDED_PATHS.forEach((path) => expandedProfilePaths.add(path));
        hasInitializedProfileTree = true;
    }

    highlightFields
        .map(normalizePath)
        .forEach((field) => {
            const parts = field.split('.');
            let currentPath = '';

            parts.forEach((part) => {
                currentPath = currentPath ? `${currentPath}.${part}` : part;
                expandedProfilePaths.add(currentPath);
            });
        });
}

function scheduleHighlightCleanup() {
    if (profileHighlightTimer) {
        clearTimeout(profileHighlightTimer);
    }

    profileHighlightTimer = setTimeout(() => {
        profileContent.querySelectorAll('.highlight').forEach((node) => {
            node.classList.remove('highlight');
        });
        profileHighlightTimer = null;
    }, PROFILE_HIGHLIGHT_DURATION);
}

async function resetChat() {
    if (!confirm('确定要清空当前对话记录吗？')) return;

    try {
        const response = await fetch(`${API_URL}/api/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ scope: 'chat' })
        });

        if (!response.ok) throw new Error('重置失败');

        clearConversationView();
        showUpdateNotification([], '对话已重置');
    } catch (error) {
        console.error('重置失败:', error);
        alert(`重置失败: ${error.message}`);
    }
}

async function resetExperimentBaseline() {
    if (isLoading) return;
    if (!confirm('确定要恢复到应用启动时的初始画像和初始记忆吗？')) return;

    try {
        const response = await fetch(`${API_URL}/api/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ scope: 'experiment' })
        });

        if (!response.ok) throw new Error('实验复位失败');

        clearConversationView();
        ablationSelectionUnlocked = true;
        ablationSelectionLocked = false;
        ablationInputs.forEach((input) => {
            input.checked = false;
        });
        syncAblationControls();
        await loadProfile();
        showUpdateNotification([], '已恢复到启动基线，可以重新选择消融维度');
    } catch (error) {
        console.error('实验复位失败:', error);
        alert(`实验复位失败: ${error.message}`);
    }
}

function clearConversationView() {
    chatHistory.innerHTML = '';
    addMessageToChat(DEFAULT_SYSTEM_MESSAGE, 'system');
}

function syncAblationControls() {
    const disableSelection = !ablationSelectionUnlocked || ablationSelectionLocked;
    ablationInputs.forEach((input) => {
        input.disabled = disableSelection;
    });

    if (clearAblationBtn) {
        clearAblationBtn.disabled = disableSelection;
    }

    updateAblationStatus();
}

function updateAblationStatus() {
    if (!ablationStatusText) return;

    if (!ablationSelectionUnlocked) {
        ablationStatusText.textContent = '先点击实验复位，再选择一个维度开始测试';
        return;
    }

    const selected = getSelectedAblationDimension();
    if (ablationSelectionLocked) {
        ablationStatusText.textContent = selected
            ? `当前实验维度：${getAblationLabel(selected)}。如需切换，请先重新复位。`
            : '当前实验配置已锁定，如需切换请先重新复位。';
        return;
    }

    ablationStatusText.textContent = selected
        ? `已选择${getAblationLabel(selected)}，发送后会锁定本轮实验配置。`
        : '已恢复到启动基线，现在可以选择一个维度开始测试。';
}

function getAblationLabel(value) {
    if (value === 'state') return '状态轴';
    if (value === 'context') return '语境轴';
    if (value === 'memory') return '时间轴';
    return '全量模式';
}

function syncExperimentToggleIcon() {
    if (!experimentPanel) return;
    const icon = experimentPanel.querySelector('.experiment-toggle-icon');
    if (icon) {
        icon.textContent = experimentPanel.open ? '−' : '+';
    }
}

function showUpdateNotification(highlightFields = [], customMessage = '') {
    const existingToast = document.querySelector('.toast');
    if (existingToast) {
        existingToast.remove();
    }

    const notification = document.createElement('div');
    notification.className = 'toast';

    const dot = document.createElement('span');
    dot.className = 'toast-dot';

    const message = document.createElement('div');
    message.textContent = customMessage || (
        highlightFields.length > 0
            ? `用户画像已更新：${highlightFields.join('、')}`
            : '用户画像已更新'
    );

    notification.appendChild(dot);
    notification.appendChild(message);
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-8px)';
        notification.style.transition = 'all 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, TOAST_DURATION);
}

function updateStatus(mode = 'idle') {
    const current = STATUS[mode] || STATUS.idle;
    statusIndicator.style.background = current.color;
    statusIndicator.style.boxShadow = `0 0 0 6px ${hexToRgba(current.color, 0.16)}`;
    statusText.textContent = current.text;
}

function hexToRgba(hex, alpha) {
    const normalized = hex.replace('#', '');
    const value = normalized.length === 3
        ? normalized.split('').map((char) => char + char).join('')
        : normalized;

    const red = Number.parseInt(value.slice(0, 2), 16);
    const green = Number.parseInt(value.slice(2, 4), 16);
    const blue = Number.parseInt(value.slice(4, 6), 16);

    return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

// ===== Voice =====
let voiceMode = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let currentAudio = null;
let ttsSentenceBuffer = '';
let voiceStartTime = null;
let voiceStopTime = null;
let firstAudioLogged = false;
let systemStartTime = null;
const SENTENCE_END_RE = /[。！？!?\n]/;
const SOFT_BREAK_RE = /[，,、；;]/;
const SOFT_BREAK_MIN_CHARS = 15;

// Pipeline state: synthesis runs in parallel up to MAX_CONCURRENT_SYNTH,
// playback consumes audioBuffer entries in seq order.
let synthEpoch = 0;
let synthSeq = 0;
let playSeq = 0;
let synthInFlight = 0;
let isPlaying = false;
const ttsStreams = new Map();
const MAX_CONCURRENT_SYNTH = 3;

function stopAudio() {
    synthEpoch++;
    if (currentAudio) { currentAudio.pause(); currentAudio = null; }
    ttsStreams.clear();
    synthSeq = playSeq = synthInFlight = 0;
    isPlaying = false;
    ttsSentenceBuffer = '';
    firstAudioLogged = false;
}

function clientLog(event, data) {
    fetch(`${API_URL}/api/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ event, ...data }),
    }).catch(() => {});
}

// ===== WebSocket voice client =====
let voiceWS = null;
let wsSeqCounter = 0;
const wsPending = new Map();

function ensureWS() {
    if (voiceWS && voiceWS.readyState === WebSocket.OPEN) return Promise.resolve(voiceWS);
    return new Promise((resolve, reject) => {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/voice`;
        voiceWS = new WebSocket(url);
        voiceWS.binaryType = 'arraybuffer';
        voiceWS._reopen = null;
        voiceWS.onopen = () => resolve(voiceWS);
        voiceWS.onerror = (e) => { reject(e); };
        voiceWS.onclose = () => {
            voiceWS = null;
            for (const [, p] of wsPending) p.reject(new Error('ws closed'));
            wsPending.clear();
        };
        voiceWS.onmessage = (e) => {
            if (typeof e.data === 'string') {
                let msg;
                try { msg = JSON.parse(e.data); } catch { return; }
                if (msg.type === 'asr_partial') { handleASRPartial(msg); return; }
                if (msg.type === 'asr_final') { handleASRFinal(msg); return; }
                if (msg.type === 'asr_empty') { handleASREmpty(msg); return; }
                if (msg.type === 'asr_end_signal') { return; }
                if (msg.type === 'asr_error') { handleASRError(msg); return; }
                if (msg.type === 'chat_start') { handleChatStart(msg); return; }
                if (msg.type === 'chat_token') { handleChatToken(msg); return; }
                if (msg.type === 'chat_done') { handleChatDone(msg); return; }
                if (msg.type === 'chat_error') { handleChatError(msg); return; }
                if (msg.type === 'tts_end') { handleTTSEnd(msg); return; }
                const p = wsPending.get(msg.seq);
                if (!p) return;
                if (msg.type === 'asr_result') {
                    wsPending.delete(msg.seq);
                    p.resolve(msg.text);
                }
            } else {
                handleTTSBinaryFrame(e.data);
            }
        };
    });
}

function blobToBase64(blob) {
    return new Promise((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result.split(',')[1]);
        r.readAsDataURL(blob);
    });
}

async function wsASR(blob) {
    const ws = await ensureWS();
    const seq = ++wsSeqCounter;
    const audio = await blobToBase64(blob);
    return new Promise((resolve, reject) => {
        wsPending.set(seq, { resolve, reject });
        ws.send(JSON.stringify({ type: 'asr', seq, audio, system_start_ms: systemStartTime }));
    });
}

async function wsTTS(text, ttsSeq) {
    const ws = await ensureWS();
    const seq = ++wsSeqCounter;
    return new Promise((resolve, reject) => {
        wsPending.set(seq, { resolve, reject });
        ws.send(JSON.stringify({ type: 'tts', seq, text, system_start_ms: systemStartTime }));
    });
}

function scheduleSynthesis(text) {
    const seq = synthSeq++;
    const epoch = synthEpoch;
    synthInFlight++;
    if (!pipeFirstTtsTriggerMs) pipeFirstTtsTriggerMs = Date.now();
    // Initialize streaming state for this seq
    if (!ttsStreams.has(seq)) ttsStreams.set(seq, { chunks: [], ended: false, epoch });
    ensureWS().then(ws => {
        if (epoch !== synthEpoch) return;
        ws.send(JSON.stringify({
            type: 'tts', seq, text,
            system_start_ms: systemStartTime,
        }));
    }).catch(() => {
        synthInFlight--;
        if (ttsStreams.has(seq)) ttsStreams.get(seq).ended = true;
        tryPlayNextStream();
    });
}

function handleTTSBinaryFrame(data) {
    const dv = new DataView(data);
    const seq = dv.getUint32(0);
    const chunk = new Uint8Array(data.slice(4));
    if (!ttsStreams.has(seq)) ttsStreams.set(seq, { chunks: [], ended: false, epoch: synthEpoch });
    const stream = ttsStreams.get(seq);
    stream.chunks.push(chunk);
    // If this seq is already playing, feed directly into the source buffer
    if (seq === playSeq && stream._flush) {
        stream._flush();
    } else {
        tryPlayNextStream();
    }
}

function handleTTSEnd(msg) {
    const seq = msg.seq;
    if (ttsStreams.has(seq)) {
        ttsStreams.get(seq).ended = true;
    } else {
        ttsStreams.set(seq, { chunks: [], ended: true, epoch: synthEpoch });
    }
    synthInFlight = Math.max(0, synthInFlight - 1);
    const stream = ttsStreams.get(seq);
    if (seq === playSeq && stream && stream._flush) {
        stream._flush();
    } else {
        tryPlayNextStream();
    }
}

function tryPlayNextStream() {
    if (isPlaying) return;
    if (!ttsStreams.has(playSeq)) {
        if (synthInFlight === 0 && playSeq > 0) {
            stopBargeInWatcher();
        }
        return;
    }
    const stream = ttsStreams.get(playSeq);
    if (stream.epoch !== synthEpoch) {
        ttsStreams.delete(playSeq);
        playSeq++;
        tryPlayNextStream();
        return;
    }
    if (stream.chunks.length === 0) {
        if (stream.ended) {
            ttsStreams.delete(playSeq);
            playSeq++;
            tryPlayNextStream();
        }
        return;
    }

    isPlaying = true;
    const curSeq = playSeq;
    const mediaSource = new MediaSource();
    const url = URL.createObjectURL(mediaSource);
    currentAudio = new Audio(url);
    let sourceBuffer = null;
    let endedLocally = false;

    const flushQueue = () => {
        if (!sourceBuffer || sourceBuffer.updating) return;
        if (stream.chunks.length > 0) {
            try {
                sourceBuffer.appendBuffer(stream.chunks.shift());
            } catch (e) {
                console.error('appendBuffer failed', e);
            }
        } else if (stream.ended && !endedLocally) {
            endedLocally = true;
            try { mediaSource.endOfStream(); } catch (e) {}
        }
    };

    mediaSource.addEventListener('sourceopen', () => {
        try {
            sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
            sourceBuffer.addEventListener('updateend', flushQueue);
            stream._flush = flushQueue;  // expose so incoming chunks can feed directly
            flushQueue();
        } catch (e) {
            console.error('addSourceBuffer failed', e);
        }
    });

    currentAudio.onplaying = () => {
        startBargeInWatcher();
        if (curSeq === 0 && !firstAudioLogged) {
            firstAudioLogged = true;
            pipeFirstAudioPlayMs = Date.now();
            emitPipelineLog();
        }
    };
    currentAudio.onended = () => {
        stream._flush = null;
        URL.revokeObjectURL(url);
        ttsStreams.delete(curSeq);
        currentAudio = null;
        playSeq++;
        isPlaying = false;
        tryPlayNextStream();
    };
    currentAudio.onerror = currentAudio.onended;
    currentAudio.play().catch(e => {
        console.error('audio play failed', e);
        if (currentAudio.onended) currentAudio.onended();
    });
}

function feedTTSToken(token) {
    ttsSentenceBuffer += token;
    // Hard stop: always break immediately
    let idx = ttsSentenceBuffer.search(SENTENCE_END_RE);
    // Soft stop: break at comma/semicolon once buffer is long enough
    if (idx === -1 && ttsSentenceBuffer.length >= SOFT_BREAK_MIN_CHARS) {
        idx = ttsSentenceBuffer.search(SOFT_BREAK_RE);
    }
    if (idx !== -1) {
        const sentence = ttsSentenceBuffer.slice(0, idx + 1).trim();
        if (sentence && synthInFlight < MAX_CONCURRENT_SYNTH) scheduleSynthesis(sentence);
        else if (sentence) {
            // backpressure: hold in buffer, retry on next pump
            ttsSentenceBuffer = sentence + ttsSentenceBuffer.slice(idx + 1);
            return;
        }
        ttsSentenceBuffer = ttsSentenceBuffer.slice(idx + 1);
    }
}

function finalizeTTS() {
    const remaining = ttsSentenceBuffer.trim();
    if (remaining && synthInFlight < MAX_CONCURRENT_SYNTH) scheduleSynthesis(remaining);
    ttsSentenceBuffer = '';
}

const micBtn = document.getElementById('micBtn');

let audioCtx = null;
let audioProcessor = null;
let currentStream = null;
let activeASRSessionId = null;
let silenceStart = null;
let voiceUserBubbleContent = null;
let asrEndSignalAt = null;
// Pipeline timing collectors (reset on each voice interaction)
let pipeSpeechStartMs = null;
let pipeSpeechEndMs = null;
let pipeAsrCompleteMs = null;
let pipeFirstTokenMs = null;
let pipeFirstTtsTriggerMs = null;
let pipeFirstAudioPlayMs = null;
const SILENCE_THRESHOLD = 0.012;
const SILENCE_DURATION_MS = 1500;

function resetPipeline() {
    pipeSpeechStartMs = null;
    pipeSpeechEndMs = null;
    pipeAsrCompleteMs = null;
    pipeFirstTokenMs = null;
    pipeFirstTtsTriggerMs = null;
    pipeFirstAudioPlayMs = null;
}

function emitPipelineLog() {
    const fmt = (a, b) => (a && b) ? Math.round(b - a) : null;
    const speech_start = pipeSpeechStartMs;
    const speech_end = pipeSpeechEndMs;
    const asr_complete = pipeAsrCompleteMs;
    const first_token = pipeFirstTokenMs;
    const tts_trigger = pipeFirstTtsTriggerMs;
    const first_audio = pipeFirstAudioPlayMs;
    clientLog('pipeline', {
        // 1. 用户开始说话
        speech_start_ms: speech_start,
        // 2. 用户说话完成
        speech_end_ms: speech_end,
        // 3. 说完 → ASR 完成
        speech_end_to_asr_complete_ms: fmt(speech_end, asr_complete),
        // 4. ASR 完成 → LLM 首 token
        asr_to_first_token_ms: fmt(asr_complete, first_token),
        // 5. LLM 首 token → TTS 触发
        first_token_to_tts_trigger_ms: fmt(first_token, tts_trigger),
        // 6. TTS 触发 → 首音播放
        tts_trigger_to_first_audio_ms: fmt(tts_trigger, first_audio),
        // 7. 总：说完 → 首音播放
        total_speech_end_to_first_audio_ms: fmt(speech_end, first_audio),
        // 辅助绝对时间戳
        asr_complete_ms: asr_complete,
        first_token_ms: first_token,
        tts_trigger_ms: tts_trigger,
        first_audio_play_ms: first_audio,
    });
}

function float32ToInt16(float32) {
    const out = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
}

function computeRMS(float32) {
    let sum = 0;
    for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
    return Math.sqrt(sum / float32.length);
}

function handleASRPartial(msg) {
    if (msg.session_id !== activeASRSessionId) return;
    const text = msg.text || '';
    userInput.value = text;
    if (!text) return;
    if (!voiceUserBubbleContent) {
        voiceUserBubbleContent = addMessageToChat(text, 'user');
    } else {
        voiceUserBubbleContent.textContent = text;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}

function handleASRFinal(msg) {
    if (msg.session_id !== activeASRSessionId) return;
    const text = msg.text || '';
    userInput.value = text;
    if (voiceUserBubbleContent && text) {
        voiceUserBubbleContent.textContent = text;
    }
    if (msg.speech_start_ms) pipeSpeechStartMs = msg.speech_start_ms;
    if (msg.speech_end_ms) pipeSpeechEndMs = msg.speech_end_ms;
    if (msg.asr_complete_ms) pipeAsrCompleteMs = msg.asr_complete_ms;
}

function handleASREndSignal(msg) {
    // Deprecated: chat is now triggered from backend on ASR completion.
    // Kept for compatibility; no-op.
}

let voiceAssistantContent = null;

function handleChatStart(msg) {
    if (isRecording) stopStreamingRecording(true);
    voiceUserBubbleContent = null;
    userInput.value = '';
    micBtn.textContent = '🎤';
    voiceMode = true;
    isLoading = true;
    sendBtn.disabled = true;
    userInput.disabled = true;
    updateStatus('loading');
    firstAudioLogged = false;
    voiceAssistantContent = addMessageToChat('', 'assistant');
}

function handleChatToken(msg) {
    const content = msg.content || '';
    if (voiceAssistantContent) {
        voiceAssistantContent.textContent += content;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    if (voiceMode) feedTTSToken(content);
    if (!pipeFirstTokenMs) pipeFirstTokenMs = Date.now();
}

function handleChatDone(msg) {
    if (voiceMode) { finalizeTTS(); voiceMode = false; }
    if (msg.response && voiceAssistantContent) {
        voiceAssistantContent.textContent = msg.response;
    }
    voiceAssistantContent = null;
    isLoading = false;
    sendBtn.disabled = false;
    userInput.disabled = false;
    updateStatus('idle');
    const highlightFields = Array.isArray(msg.updated_fields) ? msg.updated_fields : [];
    loadProfile(highlightFields).then(() => {
        if (highlightFields.length > 0) showUpdateNotification(highlightFields);
    });
}

function handleChatError(msg) {
    addMessageToChat(`错误: ${msg.error || '未知错误'}`, 'system');
    if (voiceMode) { voiceMode = false; }
    isLoading = false;
    sendBtn.disabled = false;
    userInput.disabled = false;
    updateStatus('idle');
    voiceAssistantContent = null;
}

function handleASRError(msg) {
    addMessageToChat(`语音识别出错: ${msg.error || '未知错误'}`, 'system');
    activeASRSessionId = null;
    if (isRecording) stopStreamingRecording(true);
    micBtn.textContent = '🎤';
}

function handleASREmpty(msg) {
    if (msg.session_id !== activeASRSessionId) return;
    activeASRSessionId = null;
    if (isRecording) stopStreamingRecording(true);
    voiceUserBubbleContent = null;
    userInput.value = '';
    micBtn.textContent = '🎤';
    addMessageToChat('未识别到语音，请重试', 'system');
}

function onAudioProcess(e) {
    if (!isRecording) return;
    const float32 = e.inputBuffer.getChannelData(0);
    const int16 = float32ToInt16(float32);
    if (voiceWS && voiceWS.readyState === WebSocket.OPEN) {
        voiceWS.send(int16.buffer);
    }
    const rms = computeRMS(float32);
    if (rms < SILENCE_THRESHOLD) {
        if (silenceStart === null) silenceStart = performance.now();
        else if (performance.now() - silenceStart > SILENCE_DURATION_MS) {
            silenceStart = null;
            stopStreamingRecording(false);
        }
    } else {
        silenceStart = null;
    }
}

function stopStreamingRecording(silent) {
    if (!isRecording) return;
    isRecording = false;
    micBtn.classList.remove('recording');
    if (!silent) micBtn.textContent = '⏳';
    voiceStopTime = performance.now();
    systemStartTime = voiceStopTime ? Date.now() : systemStartTime;
    if (audioProcessor) { try { audioProcessor.disconnect(); } catch {} audioProcessor = null; }
    if (audioCtx) { try { audioCtx.close(); } catch {} audioCtx = null; }
    if (currentStream) {
        currentStream.getTracks().forEach((t) => t.stop());
        currentStream = null;
    }
    if (voiceWS && voiceWS.readyState === WebSocket.OPEN && activeASRSessionId) {
        voiceWS.send(JSON.stringify({ type: 'asr_stream_end', session_id: activeASRSessionId }));
    }
}

async function startRecordingSession() {
    if (isRecording) return;
    stopBargeInWatcher();
    stopAudio();
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });
    } catch {
        alert('无法访问麦克风，请检查权限设置。');
        return;
    }
    voiceStartTime = performance.now();
    voiceStopTime = null;
    systemStartTime = Date.now();
    resetPipeline();
    try {
        const ws = await ensureWS();
        activeASRSessionId = `asr_${Date.now()}`;
        ws.send(JSON.stringify({
            type: 'asr_stream_start',
            session_id: activeASRSessionId,
            system_start_ms: systemStartTime,
            ablate_dimension: getSelectedAblationDimension(),
        }));
    } catch (e) {
        addMessageToChat(`WebSocket 连接失败: ${e.message}`, 'system');
        stream.getTracks().forEach((t) => t.stop());
        return;
    }
    currentStream = stream;
    audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(stream);
    audioProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
    silenceStart = null;
    audioProcessor.onaudioprocess = onAudioProcess;
    source.connect(audioProcessor);
    audioProcessor.connect(audioCtx.destination);
    isRecording = true;
    micBtn.classList.add('recording');
    micBtn.textContent = '⏹';
}

micBtn.addEventListener('click', async () => {
    if (isRecording) {
        stopStreamingRecording(false);
        return;
    }
    await startRecordingSession();
});

// ===== Barge-in watcher (full-duplex) =====
let bargeInStream = null;
let bargeInAudioCtx = null;
let bargeInAnalyser = null;
let bargeInRAF = null;
let bargeInActive = false;
let bargeInConsecutive = 0;
const BARGE_IN_THRESHOLD = 0.15;
const BARGE_IN_TRIGGER_CHUNKS = 3;

function startBargeInWatcher() {
    if (bargeInActive || isRecording) return;
    navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
        }
    }).then(stream => {
        if (isRecording) { stream.getTracks().forEach(t => t.stop()); return; }
        bargeInStream = stream;
        bargeInAudioCtx = new AudioContext();
        const source = bargeInAudioCtx.createMediaStreamSource(stream);
        bargeInAnalyser = bargeInAudioCtx.createAnalyser();
        bargeInAnalyser.fftSize = 1024;
        source.connect(bargeInAnalyser);
        bargeInActive = true;
        bargeInConsecutive = 0;
        const buffer = new Uint8Array(bargeInAnalyser.fftSize);
        const tick = () => {
            if (!bargeInActive) return;
            bargeInAnalyser.getByteTimeDomainData(buffer);
            let sum = 0;
            for (let i = 0; i < buffer.length; i++) {
                const v = (buffer[i] - 128) / 128;
                sum += v * v;
            }
            const rms = Math.sqrt(sum / buffer.length);
            if (rms > BARGE_IN_THRESHOLD) {
                bargeInConsecutive++;
                if (bargeInConsecutive >= BARGE_IN_TRIGGER_CHUNKS) {
                    clientLog('barge_in', { rms: rms.toFixed(3) });
                    triggerBargeIn();
                    return;
                }
            } else {
                bargeInConsecutive = 0;
            }
            bargeInRAF = requestAnimationFrame(tick);
        };
        tick();
    }).catch(e => {
        console.warn('Barge-in mic unavailable:', e);
    });
}

function stopBargeInWatcher() {
    bargeInActive = false;
    if (bargeInRAF) { cancelAnimationFrame(bargeInRAF); bargeInRAF = null; }
    if (bargeInStream) {
        bargeInStream.getTracks().forEach(t => t.stop());
        bargeInStream = null;
    }
    if (bargeInAudioCtx) {
        try { bargeInAudioCtx.close(); } catch {}
        bargeInAudioCtx = null;
    }
    bargeInAnalyser = null;
    bargeInConsecutive = 0;
}

async function triggerBargeIn() {
    stopBargeInWatcher();
    stopAudio();
    if (currentAudio) { try { currentAudio.pause(); } catch {} currentAudio = null; }
    addMessageToChat('-- 已打断 --', 'system');
    await startRecordingSession();
}
