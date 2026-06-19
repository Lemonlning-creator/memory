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
const STATIC_PROFILE_LAYER_ORDER = ['core', 'regulation', 'cognitive_style', 'behavior_preference', 'social_physical'];
const DEFAULT_EXPANDED_PATHS = [
    'state_axis',
    'state_axis.static_profile',
    'state_axis.current_state',
    'state_axis.static_profile.core',
    'state_axis.static_profile.social_physical',
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
let selectedCharacterId = null;
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
    if (!message || isLoading) return;
    if (!selectedCharacterId) {
        showUpdateNotification([], 'Please select a character first.');
        return;
    }

    const ablateDimension = getSelectedAblationDimension();

    isLoading = true;
    sendBtn.disabled = true;
    userInput.disabled = true;
    updateStatus('loading');

    try {
        addMessageToChat(message, 'user');
        userInput.value = '';

        const assistantContent = addMessageToChat('', 'assistant');

        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                character_id: selectedCharacterId,
                ablate_dimension: ablateDimension
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '请求失败');
        }

        const data = await readChatStream(response, assistantContent);
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

async function initializeCharacterCards() {
    try {
        const response = await fetch(`${API_URL}/api/characters`);
        if (!response.ok) throw new Error('failed to load characters');

        const data = await response.json();
        renderCharacterCards(data.characters || [], data.active_character_id || null);
    } catch (error) {
        console.error('load characters failed:', error);
        if (characterCards) {
            characterCards.innerHTML = '<div class="loading" style="color:#d85d59;">Failed to load characters</div>';
        }
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

async function selectCharacter(characterId) {
    if (!characterId || isLoading) return;

    try {
        setChatEnabled(false);
        const response = await fetch(`${API_URL}/api/characters/select`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character_id: characterId })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'failed to select character');
        }

        selectedCharacterId = characterId;
        document.body.classList.remove('character-picking');
        document.querySelectorAll('.character-card').forEach((card) => {
            card.classList.toggle('active', card.dataset.characterId === characterId);
        });

        clearConversationView();
        renderProfile(data.profile || {}, []);
        setChatEnabled(true);
        showUpdateNotification([], `${data.character.display_name || data.character.name} is ready.`);
    } catch (error) {
        console.error('select character failed:', error);
        showUpdateNotification([], `Character load failed: ${error.message}`);
    }
}

function setChatEnabled(enabled) {
    userInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    if (!enabled && !selectedCharacterId) {
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

async function readChatStream(response, contentNode) {
    if (!response.body) {
        const data = await response.json();
        contentNode.textContent = data.message || '';
        return data;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalData = null;

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
                contentNode.textContent += event.content || '';
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
