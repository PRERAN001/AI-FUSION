from flask import Flask, request, jsonify, render_template_string
import requests
from groq import Groq
import joblib
import requests


app = Flask(__name__)

# Load pre-trained models and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
reverse_label_mapping = joblib.load('reverse_label_mapping.pkl')

# Define API functions with error handling
def deepseekk(prompt: str, model: str = "deepseek-r1-distill-llama-70b"):
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
    except Exception as e:
        print(f"Error streaming Groq response: {e}")
        return None

#done
def stream_groq_response(prompt: str, model: str = "mistral-saba-24b"):
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
    except Exception as e:
        print(f"Error streaming Groq response: {e}")
        return None

def fetch_from_gemini_flash(prompt: str):
    try:
        api_key = "AIzaSyC4ycxtKj51xjzdrachRUKImGbJX55Gwv8"  # 🔐 Replace with your actual API key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print(f"Error fetching from Gemini Flash: {e}")
        return None

def chat_with_gemma(prompt: str, stream: bool = False):
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=stream,
            stop=None,
        )
        if stream:
            return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
        else:
            return completion.choices[0].message.content
    except Exception as e:
        print(f"Error chatting with Gemma: {e}")
        return None

def chat_with_llama_scout(prompt: str):
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
    except Exception as e:
        print(f"Error chatting with LLaMA Scout: {e}")
        return None

def deepseek_vo(prompt: str, model: str = "deepseek-r1-distill-llama-70b"):
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"""
You are VO, a highly skilled virtual operator trained to instantly generate modern, production-ready web development code.

Your job is to:
- Accept user input in plain English
- Interpret it clearly
- Output only the requested HTML, CSS, JavaScript, React, or Tailwind code
- Do not include explanations, just the code block

Prompt: {prompt}
"""}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
    except Exception as e:
        print(f"Error streaming Groq response: {e}")
        return None

def stream_from_qwen(prompt: str, temperature: float = 0.6, max_tokens: int = 4096, top_p: float = 0.95):
    try:
        client = Groq(api_key="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")
        completion = client.chat.completions.create(
            model="qwen-qwq-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            stop=None,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in completion)
    except Exception as e:
        print(f"Error streaming from Qwen: {e}")
        return None

# Determine lead model based on predicted category
def get_lead_model(predicted_category):
    if predicted_category == "Code generation":
        return "deepseek"
    elif predicted_category in ["Summarizing search results", "Explanation of complex concepts"]:
        return "llama"
    elif predicted_category == "Generating production-ready React + Tailwind UI code":
        return "v0"
    elif predicted_category == "Quick fact-based answers":
        return "gemini"
    elif predicted_category in ["Mathematical computation", "Deep document understanding"]:
        return "qwen"
    elif predicted_category in ["Logical reasoning", "Ethical reasoning"]:
        return "gemma"
    return None

# # Model pipeline
model_map = {
    "code generation": [deepseekk, stream_from_qwen, chat_with_gemma, stream_groq_response],
    "creative writing": [deepseekk, stream_groq_response, chat_with_gemma, chat_with_llama_scout],
    "logical reasoning": [chat_with_gemma, deepseekk, stream_from_qwen, stream_groq_response],
    "deep document understanding": [chat_with_llama_scout, deepseekk, fetch_from_gemini_flash, stream_groq_response],
    "explanation of complex concepts": [fetch_from_gemini_flash, deepseekk, chat_with_llama_scout, chat_with_gemma],
    "long-context summarization": [chat_with_llama_scout, deepseekk, fetch_from_gemini_flash, stream_groq_response],
    "ethical reasoning": [chat_with_gemma, deepseekk, fetch_from_gemini_flash, stream_groq_response],
    "web-integrated answers": [stream_groq_response, deepseekk, fetch_from_gemini_flash, chat_with_llama_scout],
    "search-based reasoning": [stream_groq_response, deepseekk, fetch_from_gemini_flash, chat_with_llama_scout],
    "summarizing search results": [chat_with_llama_scout, fetch_from_gemini_flash, deepseekk, stream_groq_response],
    "citing web sources": [deepseekk, fetch_from_gemini_flash, chat_with_llama_scout, stream_groq_response],
    "quick fact-based answers": [fetch_from_gemini_flash, deepseekk, stream_groq_response, chat_with_gemma],
    "generating production-ready react + tailwind ui code": [deepseek_vo, stream_groq_response, chat_with_gemma, stream_from_qwen],
    "mathematical computation": [stream_from_qwen, deepseekk, chat_with_gemma, stream_groq_response]
}

def process_query(text):
    my_input_tfidf = vectorizer.transform([text])
    prediction = model.predict(my_input_tfidf)
    predicted_category = reverse_label_mapping[prediction[0]]
    print("Predicted label:", predicted_category)

    # Normalize predicted_category to lowercase for model_map lookup
    normalized_category = predicted_category.lower()
    models = model_map.get(normalized_category)
    if not models:
        return {"error": f"Unknown category: {predicted_category}. No models found."}, None, None, None, None, None, None, None, [], None

    lead_model = get_lead_model(predicted_category)
    instructions = {
        "lead": f"Answer the user's question accurately. You are the primary model. Use your full capabilities to generate the best output. Here is the question from user: {text}",
        "lead1": "",
        "lead2": "",
        "lead3": ""
    }

    responses = {}
    lead_input = instructions["lead"]
    r1 = models[0](lead_input)
    responses["lead"] = r1

    if r1 is not None:
        instructions["lead1"] = (
            f"Review the output from the lead model and the original user question. "
            f"Improve or refine the response if needed. "
            f"Here is the output from the previous model: {r1} and here is the question asked: {text}"
        )
        r2 = models[1](instructions["lead1"])
        responses["lead1"] = r2

        if r2 is not None:
            instructions["lead2"] = (
                f"Consider the user's question and outputs from both the lead and lead1 models. "
                f"Add depth, fill in gaps, or correct mistakes. "
                f"Here is the question: {text}, lead output: {r1}, and lead1 output: {r2}"
            )
            r3 = models[2](instructions["lead2"])
            responses["lead2"] = r3

            if r3 is not None:
                instructions["lead3"] = (
                    f"Evaluate previous outputs (lead, lead1, lead2) along with the original question. "
                    f"Provide final checks, enhancements, or alternative perspectives. "
                    f"Here is the question: {text}, lead output: {r1}, lead1 output: {r2}, and lead2 output: {r3}"
                )
                lead3 = models[3](instructions["lead3"])
                responses["lead3"] = lead3

    # Extract AI contributors from used models
    ai_contributors = []
    if r1 and "deepseekk" in str(models[0]):
        ai_contributors.append({"name": "deepseekk", "specialty": "Language", "color": "bg-gray-600"})
    if r1 and "fetch_from_gemini_flash" in str(models[0]):
        ai_contributors.append({"name": "Gemini", "specialty": "Multimodal", "color": "bg-gray-700"})
    if r2 and "chat_with_llama_scout" in str(models[1]):
        ai_contributors.append({"name": "LLaMA Scout", "specialty": "Summarization", "color": "bg-gray-600"})
    if r3 and "chat_with_gemma" in str(models[2]):
        ai_contributors.append({"name": "Gemma", "specialty": "Reasoning", "color": "bg-gray-700"})
    if any("stream_groq_response" in str(m) for m in models):
        ai_contributors.append({"name": "Grok", "specialty": "Reasoning", "color": "bg-gray-600"})
    if any("fetch_from_v0_sim" in str(m) for m in models):
        ai_contributors.append({"name": "V0 Sim", "specialty": "UI Coding", "color": "bg-gray-700"})
    if any("stream_from_qwen" in str(m) for m in models):
        ai_contributors.append({"name": "Qwen", "specialty": "Computation", "color": "bg-gray-600"})

    # Clean up special characters in response values
    import string
    special_chars = ['!', '"', '#', '$', '%', '&', "'", '-', '.', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    for key in responses:
        if responses[key]:
            for char in special_chars:
                responses[key] = responses[key].replace(char, "")

    return responses["lead"], responses["lead1"], responses["lead2"], responses["lead3"], models[0], models[1], models[2], models[3], ai_contributors, lead_model
# Flask routes
@app.route('/')
def index():
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI FUSION</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
            overflow-x: hidden;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 1px solid #333;
            padding-bottom: 30px;
        }

        .logo {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #ffffff, #888);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            letter-spacing: 2px;
        }

        .tagline {
            font-size: 1.2rem;
            color: #aaa;
            margin-bottom: 20px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 40px;
            margin-bottom: 40px;
        }

        .input-section {
            background: #111;
            border-radius: 15px;
            padding: 30px;
            border: 1px solid #222;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .input-section h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #fff;
        }

        .input-group {
            margin-bottom: 25px;
        }

        textarea {
            width: 100%;
            min-height: 150px;
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 15px;
            color: #fff;
            font-size: 16px;
            resize: vertical;
            transition: border-color 0.3s ease;
        }

        textarea:focus {
            outline: none;
            border-color: #555;
            box-shadow: 0 0 10px rgba(255,255,255,0.1);
        }

        .submit-btn {
            width: 100%;
            background: linear-gradient(135deg, #333, #555);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .submit-btn:hover {
            background: linear-gradient(135deg, #555, #777);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255,255,255,0.2);
        }

        .submit-btn:disabled {
            background: #222;
            cursor: not-allowed;
            transform: none;
        }

        .results-section {
            background: #111;
            border-radius: 15px;
            padding: 30px;
            border: 1px solid #222;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .ai-contributors {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 25px;
            padding: 15px;
            background: #0a0a0a;
            border-radius: 10px;
            border: 1px solid #333;
        }

        .ai-chip {
            background: #333;
            color: #fff;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid #555;
        }

        .response-tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
        }

        .tab-btn {
            background: none;
            border: none;
            color: #aaa;
            padding: 15px 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }

        .tab-btn.active {
            color: #fff;
            border-bottom-color: #555;
        }

        .tab-btn:hover {
            color: #fff;
            background: #1a1a1a;
        }

        .response-content {
            display: none;
            background: #0a0a0a;
            border-radius: 10px;
            padding: 25px;
            border: 1px solid #333;
            min-height: 200px;
        }

        .response-content.active {
            display: block;
        }

        .response-text {
            font-size: 15px;
            line-height: 1.8;
            color: #e0e0e0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .loading-spinner {
            border: 3px solid #333;
            border-top: 3px solid #fff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .category-display {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            text-align: center;
        }

        .category-label {
            font-size: 14px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        .category-value {
            font-size: 18px;
            font-weight: 600;
            color: #fff;
        }

        .model-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .model-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }

        .model-name {
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        .model-value {
            font-size: 14px;
            font-weight: 600;
            color: #fff;
        }

        .empty-state {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 40px;
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .logo {
                font-size: 2.5rem;
            }
            
            .container {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1 class="logo">AI FUSION</h1>
            <p class="tagline">Multi-Model AI Intelligence System</p>
            <p style="color: #666; font-size: 14px;">Combining specialized AI models for optimal task-specific solutions</p>
        </header>

        <div class="main-content">
            <div class="input-section">
                <h2>Ask Your Question</h2>
                <div class="input-group">
                    <textarea 
                        id="queryInput" 
                        placeholder="Enter your question or task here... The system will automatically select the best AI models for your specific needs."
                    ></textarea>
                </div>
                <button class="submit-btn" id="submitBtn" onclick="processQuery()">
                    Analyze & Process
                </button>
            </div>

            <div class="results-section">
                <div id="loadingState" class="loading">
                    <div class="loading-spinner"></div>
                    <p>AI models are processing your query...</p>
                </div>

                <div id="emptyState" class="empty-state">
                    <p>Submit a query to see AI-powered results</p>
                </div>

                <div id="resultsContent" style="display: none;">
                    <div class="category-display">
                        <div class="category-label">Detected Category</div>
                        <div class="category-value" id="categoryValue">-</div>
                    </div>

                    <div class="ai-contributors" id="aiContributors">
                        <!-- AI contributor chips will be populated here -->
                    </div>

                    <div class="model-info" id="modelInfo">
                        <!-- Model information cards will be populated here -->
                    </div>

                    <div class="response-tabs">
                        <button class="tab-btn active" onclick="showTab('lead')">Lead Response</button>
                        <button class="tab-btn" onclick="showTab('lead1')">Refinement</button>
                        <button class="tab-btn" onclick="showTab('lead2')">Enhancement</button>
                        <button class="tab-btn" onclick="showTab('lead3')">Final Check</button>
                    </div>

                    <div id="lead" class="response-content active">
                        <div class="response-text" id="leadResponse">-</div>
                    </div>
                    <div id="lead1" class="response-content">
                        <div class="response-text" id="lead1Response">-</div>
                    </div>
                    <div id="lead2" class="response-content">
                        <div class="response-text" id="lead2Response">-</div>
                    </div>
                    <div id="lead3" class="response-content">
                        <div class="response-text" id="lead3Response">-</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.response-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        async function processQuery() {
            const query = document.getElementById('queryInput').value.trim();
            
            if (!query) {
                alert('Please enter a question or task.');
                return;
            }

            // Show loading state
            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('resultsContent').style.display = 'none';
            document.getElementById('loadingState').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Update UI with results
                document.getElementById('categoryValue').textContent = data.category;
                
                // Update model info
                const modelNames = data.model_names || ['Model 1', 'Model 2', 'Model 3', 'Model 4'];
                const modelInfoHTML = modelNames.map((model, index) => `
                    <div class="model-card">
                        <div class="model-name">Model ${index + 1}</div>
                        <div class="model-value">${model}</div>
                    </div>
                `).join('');
                document.getElementById('modelInfo').innerHTML = modelInfoHTML;

                // Update AI contributors
                const contributorsHTML = data.ai_contributors.map(contributor => `
                    <div class="ai-chip">${contributor.name}</div>
                `).join('');
                document.getElementById('aiContributors').innerHTML = contributorsHTML;

                // Update responses
                document.getElementById('leadResponse').textContent = data.responses.lead || 'No response from lead model';
                document.getElementById('lead1Response').textContent = data.responses.lead1 || 'No response from lead1 model';
                document.getElementById('lead2Response').textContent = data.responses.lead2 || 'No response from lead2 model';
                document.getElementById('lead3Response').textContent = data.responses.lead3 || 'No response from lead3 model';

                // Show results
                document.getElementById('loadingState').style.display = 'none';
                document.getElementById('resultsContent').style.display = 'block';

                // Reset to first tab
                showTab('lead');
                document.querySelector('.tab-btn').classList.add('active');

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while processing your query.');
            } finally {
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('loadingState').style.display = 'none';
            }
        }

        // Enable Enter key submission
        document.getElementById('queryInput').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                processQuery();
            }
        });

        // Initialize first tab as active
        document.addEventListener('DOMContentLoaded', function() {
            showTab('lead');
        });
    </script>
</body>
</html>'''
    return render_template_string(html_template)

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        print("======================")

        # Process the query using your existing function
        actual_response_l, actual_response_l1, actual_response_l2, actual_response_l3, m0, m1, m2, m3, ai_contributors, lead_model = process_query(query)
        
        print(actual_response_l,actual_response_l1,actual_response_l2,actual_response_l3, m0, m1, m2, m3, ai_contributors, lead_model)
        # Get category prediction
        my_input_tfidf = vectorizer.transform([query])
        prediction = model.predict(my_input_tfidf)
        predicted_category = reverse_label_mapping[prediction[0]]
        
        # Get model names for display
        model_names = []
        if m0: model_names.append(str(m0.__name__) if hasattr(m0, '__name__') else 'Model 1')
        if m1: model_names.append(str(m1.__name__) if hasattr(m1, '__name__') else 'Model 2')
        if m2: model_names.append(str(m2.__name__) if hasattr(m2, '__name__') else 'Model 3')
        if m3: model_names.append(str(m3.__name__) if hasattr(m3, '__name__') else 'Model 4')
        
        return jsonify({
            'category': predicted_category,
            'lead_model': lead_model,
            'responses': {
                'lead': actual_response_l,
                'lead1': actual_response_l1,
                'lead2': actual_response_l2,
                'lead3': actual_response_l3
            },
            'model_names': model_names,
            'ai_contributors': ai_contributors
        })
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)