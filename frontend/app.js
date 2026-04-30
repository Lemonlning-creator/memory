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
const PROFILE_LABELS = {
    stable_profile: '稳定画像',
    dynamic_state: '近期状态',
    basic_info: '基本信息',
    preferences: '偏好设置'
};
const ROOT_PROFILE_ORDER = ['stable_profile', 'dynamic_state'];
const DEFAULT_EXPANDED_PATHS = [
    'stable_profile',
    'stable_profile.basic_info',
    'stable_profile.preferences',
    'dynamic_state'
];

const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistory = document.getElementById('chatHistory');
const profileContent = document.getElementById('profileContent');
const resetBtn = document.getElementById('resetBtn');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');

let isLoading = false;
let hasInitializedProfileTree = false;
let profileHighlightTimer = null;
const expandedProfilePaths = new Set(DEFAULT_EXPANDED_PATHS);

document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    updateStatus('idle');

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !isLoading) {
            sendMessage();
        }
    });
    resetBtn.addEventListener('click', resetChat);
});

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;
    userInput.disabled = true;
    updateStatus('loading');

    try {
        addMessageToChat(message, 'user');
        userInput.value = '';

        const response = await fetch(`${API_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '请求失败');
        }

        const data = await response.json();
        addMessageToChat(data.message, 'assistant');

        const highlightFields = Array.isArray(data.updated_fields) ? data.updated_fields : [];
        if (highlightFields.length > 0) {
            await loadProfile(highlightFields);
            showUpdateNotification(highlightFields);
        }
    } catch (error) {
        console.error('错误:', error);
        addMessageToChat(`错误: ${error.message}`, 'system');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        userInput.disabled = false;
        updateStatus('idle');
        userInput.focus();
    }
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

function createProfileNode(label, value, path, highlightFields = []) {
    if (!hasDisplayableContent(value)) {
        return null;
    }

    if (Array.isArray(value) && value.every(isPrimitiveValue)) {
        return createTagGroupNode(label, value, path, highlightFields);
    }

    if (Array.isArray(value) || isPlainObject(value)) {
        return createBranchNode(label, value, path, highlightFields, false);
    }

    return createLeafNode(label, value, path, highlightFields);
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
    arrow.textContent = expanded ? '⌄' : '›';

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
        arrow.textContent = nextExpanded ? '⌄' : '›';
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

function createLeafNode(label, value, path, highlightFields = []) {
    const node = document.createElement('div');
    node.className = 'profile-tree-node profile-tree-leaf';
    node.dataset.path = path;

    if (shouldHighlightPath(path, highlightFields)) {
        node.classList.add('highlight');
    }

    const row = document.createElement('div');
    row.className = 'profile-tree-row';
    if (path.startsWith('stable_profile.basic_info.')) {
        row.classList.add('profile-tree-row-inline');
    }

    const labelEl = document.createElement('span');
    labelEl.className = 'profile-tree-key';
    labelEl.textContent = label;

    const valueEl = document.createElement('div');
    valueEl.className = 'profile-tree-value';
    valueEl.textContent = formatLeafValue(value);

    row.appendChild(labelEl);
    row.appendChild(valueEl);
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

    return Object.entries(value).map(([key, itemValue]) => ({
        label: getLabelForKey(key),
        value: itemValue,
        path: joinPath(parentPath, key)
    }));
}

function getOrderedRootEntries(profile) {
    const priority = ROOT_PROFILE_ORDER
        .filter((key) => Object.hasOwn(profile, key))
        .map((key) => [key, profile[key]]);
    const remainder = Object.entries(profile)
        .filter(([key]) => !ROOT_PROFILE_ORDER.includes(key));

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
    if (!confirm('确定要重置对话历史吗？')) return;

    try {
        const response = await fetch(`${API_URL}/api/reset`, {
            method: 'POST'
        });

        if (!response.ok) throw new Error('重置失败');

        chatHistory.innerHTML = '';
        addMessageToChat('对话已重置。你好！我是你的个性化助手，请尽情聊天吧！', 'system');
        showUpdateNotification([], '对话已重置');
    } catch (error) {
        console.error('重置失败:', error);
        alert(`重置失败: ${error.message}`);
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
