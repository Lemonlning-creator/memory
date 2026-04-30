from flask import Flask, request, jsonify
from flask_cors import CORS
from agent import UserProfileAgent
import json

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# 全局Agent实例
agent = UserProfileAgent()

@app.route('/')
def index():
    """提供前端页面"""
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天消息"""
    data = request.json
    user_input = data.get('message', '').strip()
    
    if not user_input:
        return jsonify({'error': '消息不能为空'}), 400
    
    try:
        # 生成回复
        response = agent.generate_response(user_input)
        
        # 更新对话历史
        agent.conversation_history.append({"role": "user", "content": user_input})
        agent.conversation_history.append({"role": "assistant", "content": response})
        
        # 分析并更新画像
        updated_fields = agent.analyze_and_update_profile(user_input, response)
        
        return jsonify({
            'message': response,
            'profile': agent.profile,
            'conversation_length': len(agent.conversation_history),
            'updated_fields': updated_fields
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
def get_profile():
    """获取用户画像"""
    return jsonify(agent.profile), 200

@app.route('/api/profile', methods=['POST'])
def update_profile():
    """手动更新用户画像"""
    data = request.json
    try:
        agent.apply_updates(data)
        agent.save_profile()
        return jsonify({'message': '画像已更新', 'profile': agent.profile}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取对话历史"""
    return jsonify({
        'history': agent.conversation_history,
        'total': len(agent.conversation_history)
    }), 200

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """重置对话历史"""
    agent.conversation_history = []
    return jsonify({'message': '对话已重置'}), 200

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)