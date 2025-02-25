import streamlit.components.v1 as components
import json

def create_quiz_component(quiz_data, question_type="multiple_choice"):
    """Create a quiz component that integrates React with Streamlit."""
    
    # Convert quiz data to JSON for React
    quiz_json = json.dumps(quiz_data)
    
    # Define the React component HTML with dark theme support
    component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/6.26.0/babel.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #0e1117;
                color: white;
            }}
            .question-card {{
                background-color: #0e1117;
                border: 1px solid #1e1e1e;
            }}
            .radio-label {{
                color: #fafafa;
            }}
            .text-input {{
                background-color: #262730;
                color: white;
                border: 1px solid #1e1e1e;
                padding: 8px;
                width: 100%;
                border-radius: 4px;
            }}
            .submit-btn {{
                background-color: #1e1e1e;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
            }}
            .submit-btn:hover {{
                background-color: #2e2e2e;
            }}
        </style>
    </head>
    <body>
        <div id="root"></div>
        <script type="text/babel">
            const quizData = {quiz_json};
            const questionType = "{question_type}";
            
            function Quiz() {{
                const [answers, setAnswers] = React.useState({{}});
                const [feedback, setFeedback] = React.useState({{}});
                const [score, setScore] = React.useState(0);
                
                const checkAnswer = (questionIndex, selectedAnswer) => {{
                    if (questionType === "multiple_choice") {{
                        const question = quizData.questions[questionIndex];
                        const isCorrect = selectedAnswer === question.correct_answer;
                        
                        setAnswers(prev => ({{
                            ...prev,
                            [questionIndex]: selectedAnswer
                        }}));
                        
                        setFeedback(prev => ({{
                            ...prev,
                            [questionIndex]: {{
                                isCorrect,
                                explanation: question.explanation
                            }}
                        }}));
                        
                        const newAnswers = {{ ...answers, [questionIndex]: selectedAnswer }};
                        const newScore = quizData.questions.reduce((acc, q, idx) => {{
                            return acc + (newAnswers[idx] === q.correct_answer ? 1 : 0);
                        }}, 0);
                        setScore(newScore);
                    }}
                }};
                
                const submitOpenEnded = (questionIndex, answer) => {{
                    if (!answer.trim()) return;
                    
                    setAnswers(prev => ({{
                        ...prev,
                        [questionIndex]: answer
                    }}));
                    
                    const question = quizData.questions[questionIndex];
                    setFeedback(prev => ({{
                        ...prev,
                        [questionIndex]: {{
                            feedback: "Answer submitted. Compare with model answer below:",
                            modelAnswer: question.model_answer,
                            keyPoints: question.key_points
                        }}
                    }}));
                }};
                
                return (
                    <div className="p-4 text-white">
                        <h2 className="text-2xl font-bold mb-6">{{quizData.topic}} Quiz</h2>
                        
                        {{quizData.questions.map((question, index) => (
                            <div key={{index}} className="mb-8 p-4 question-card rounded-lg">
                                <h3 className="text-lg font-semibold mb-3">Question {{index + 1}}</h3>
                                <p className="mb-4">{{question.question}}</p>
                                
                                {{questionType === "multiple_choice" ? (
                                    <div className="space-y-3">
                                        {{question.options.map((option, optIndex) => (
                                            <label key={{optIndex}} className="flex items-center space-x-2 cursor-pointer radio-label">
                                                <input
                                                    type="radio"
                                                    name={{`question-${{index}}`}}
                                                    value={{option}}
                                                    checked={{answers[index] === option}}
                                                    onChange={{() => checkAnswer(index, option)}}
                                                    className="h-4 w-4"
                                                />
                                                <span className="text-sm">{{option}}</span>
                                            </label>
                                        ))}}
                                    </div>
                                ) : (
                                    <div>
                                        <textarea
                                            className="text-input mb-2"
                                            rows="4"
                                            placeholder="Type your answer here..."
                                            value={{answers[index] || ''}}
                                            onChange={{(e) => setAnswers(prev => ({{...prev, [index]: e.target.value}}))}}>
                                        </textarea>
                                        <button
                                            className="submit-btn"
                                            onClick={{() => submitOpenEnded(index, answers[index])}}>
                                            Submit Answer
                                        </button>
                                    </div>
                                )}}
                                
                                {{feedback[index] && (
                                    <div className="mt-4">
                                        {{questionType === "multiple_choice" ? (
                                            <div className={{`p-3 rounded ${{
                                                feedback[index].isCorrect ? 'bg-green-900' : 'bg-red-900'
                                            }}`}}>
                                                <p className="font-semibold">
                                                    {{feedback[index].isCorrect ? '✓ Correct!' : '✗ Incorrect'}}
                                                </p>
                                                <p className="mt-2 text-sm">{{feedback[index].explanation}}</p>
                                            </div>
                                        ) : (
                                            <div className="p-3 rounded bg-gray-800">
                                                <p className="font-semibold">{{feedback[index].feedback}}</p>
                                                <p className="mt-2 text-sm">Model Answer: {{feedback[index].modelAnswer}}</p>
                                                <p className="mt-2 text-sm">Key Points to Cover:</p>
                                                <ul className="list-disc pl-5 mt-1">
                                                    {{feedback[index].keyPoints.map((point, idx) => (
                                                        <li key={{idx}} className="text-sm">{{point}}</li>
                                                    ))}}
                                                </ul>
                                            </div>
                                        )}}
                                    </div>
                                )}}
                            </div>
                        ))}}
                        
                        {{questionType === "multiple_choice" && Object.keys(answers).length > 0 && (
                            <div className="mt-6 p-4 question-card rounded-lg">
                                <h3 className="font-semibold mb-2">Current Score</h3>
                                <p>{{score}} / {{quizData.questions.length}} correct</p>
                            </div>
                        )}}
                    </div>
                );
            }}
            
            ReactDOM.render(<Quiz />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """
    
    # Render the component
    components.html(component_html, height=600, scrolling=True)