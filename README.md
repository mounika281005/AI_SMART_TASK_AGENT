# 🧠 AI Smart Task Agent – Smart Task Manager Using LangChain & Gemini 

> Welcome to AI Task Genius Agent – a personal productivity assistant built with LangChain, Streamlit, and Google Gemini. This project leverages ReAct-based agents to understand, prioritize, and manage tasks using natural language.

---

## 🎯 Learning Objectives

This project helped me understand:

- How ReAct-based LangChain agents reason and operate step-by-step
- Integrating custom tools into AI agents
- Handling loops, fallback mechanisms, and agent failures gracefully
- Connecting LLMs (Gemini) with real-world productivity tools
- Building an interactive frontend using Streamlit

---

## ✨ Features

- **Conversational Task Handling:** Add, delete, update, and view tasks via natural language commands.
- **Interactive Task Cards:** Visually intuitive task layouts with priority, category, and deadline indicators.
- **AI-Based Suggestions:** Get intelligent advice on what to focus on next.
- **Advanced Search & Filters:** Find tasks using deadlines, effort level, categories, tags, or keywords.
- **Time Logging:** Track time spent on tasks for productivity insights.
- **Productivity Analytics:** View statistics on completed tasks, overdue tasks, and average completion time.
- **Duplicate Prevention:** The system intelligently checks for and warns about duplicate tasks.
- **Customizable UI:** Rich custom CSS for an enhanced user experience, including animations and themes.
- **Robust Agent Backend:** Utilizes a Langchain ReAct agent with custom tools, error handling, and loop prevention mechanisms.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend AI Agent:**
  - Langchain (Agents, Tools, Prompts)
  - Google Gemini (via `langchain-google-genai`)
- **Data Persistence:** JSON files (for tasks, analytics, templates)
- **Date Parsing:** `dateparser`, `python-dateutil`
- **Environment Management:** `python-dotenv`

---

## 📁 Project Structure

├── .gitignore
├── task_manager_app.py         # Streamlit frontend application
├── task_prioritizer_agent.py   # Langchain agent, tools, and task manager logic
├── tasks.json                  # Stores task data (example provided)
├── test_task_agent.py          # Test suite for the agent
├── .env.example                # Example environment file
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file

---

## Setup & Installation 🚀

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/ai-smart-task-agent.git
    cd ai-smart-task-agent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Rename `.env` to `.env`.
    *   Open `.env` and add your Google API Key:
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"
        ```
    *   You can obtain a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🏃‍♂️ How to Run

1.  **Run the Streamlit application:**
    ```bash
    streamlit run task_manager_app.py
    ```
    The application will open in your web browser.

2.  **Run the test suite (optional):**
    ```bash
    python test_task_agent.py
    ```

---

## 🎬 Example Usage

**Sample Commands:**
*   "Add task: Review project proposal by next Friday 📝"
*   "Show me all my tasks 📋"
*   "What should I work on next? 🤔"
*   "Track 30 minutes on task [ID] ⏱️"
*   "Delete task: Call John"
*   "Add tasks: 1. Design new logo by Wednesday. 2. Send client update by EOD. 3. Research competitors next week."

---

## 💡 Known Issues / Future Improvements

### Current Limitations
*   **UI Rendering & Agent Output:** The app uses structured JSON for cards and natural text for summaries. Future work will enhance consistency.
*   **Suggestion Chips Functionality:** The suggestion chips displayed below the chat input are currently visual only. Clicking them does not yet populate the input field; this JavaScript functionality needs to be fully implemented or debugged.
*   **Concurrency:** The JSON file-based data storage is not suitable for concurrent multi-user access. A database backend would be needed for such scenarios.
*   **Advanced Analytics:** More sophisticated analytics and visualizations could be added.
*   **Full CRUD for Templates:** Currently, templates are basic. Full management could be implemented.

---

## 🚀 Planned Features

- 🎤 **Voice Commands** – via Web Speech API or OpenAI Whisper  
- 📅 **Calendar Integration** – Google, Outlook sync and smart scheduling  
- 🔄 **Real-time DB Sync** – Use PostgreSQL or MongoDB  
- 📱 **Mobile PWA Support**  
- 🤝 **Multi-user Collaboration**  
- 📊 **Advanced ML-based Analytics**  
- 🔗 **Integrations** – Slack, Discord, webhooks, email  

---

## 🧠 What I Learned

- ReAct agents "think" step-by-step before acting  
- Prompt engineering directly affects LLM performance  
- Streamlit makes it easy to build powerful UI fast  
- Managing structured vs unstructured data flow is key  
- Building agents for real productivity use cases is totally achievable  

---

## 🤝 Contributing

Pull requests, bug reports, and ideas are welcome! Check out the [issues page](https://github.com/SaKinLord/ai-task-genius-agent/issues).

---

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
