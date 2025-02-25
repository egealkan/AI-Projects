# ## works without the results

# import streamlit as st


# def display_quiz_in_container(quiz_data, question_type="multiple_choice"):
#     """Display quiz in a container to prevent disappearing."""
    
#     quiz_container = st.container()
    
#     with quiz_container:
#         if question_type == "multiple_choice":
#             st.markdown(f"## {quiz_data['topic']} Quiz")
            
#             # Create form to batch all inputs
#             with st.form(key=f"quiz_form_{quiz_data['topic']}"):
#                 answers = {}
#                 for i, question in enumerate(quiz_data['questions']):
#                     st.markdown(f"### Question {i+1}")
#                     st.write(question['question'])
                    
#                     # Radio buttons inside form
#                     answer = st.radio(
#                         "Select your answer:",
#                         options=question['options'],
#                         key=f"q_{i}",
#                         index=None  # Prevents default selection
#                     )
#                     answers[i] = answer
#                     st.markdown("---")
                
#                 # Submit button at the end of form
#                 submitted = st.form_submit_button("Submit Quiz")
                
#                 if submitted:
#                     score = 0
#                     for i, question in enumerate(quiz_data['questions']):
#                         if answers[i] == question['correct_answer']:
#                             score += 1
#                             st.success(f"Question {i+1}: Correct!")
#                         else:
#                             st.error(f"Question {i+1}: Incorrect. The correct answer is: {question['correct_answer']}")
#                             st.info(f"Explanation: {question['explanation']}")
                    
#                     total = len(quiz_data['questions'])
#                     percentage = (score / total) * 100
                    
#                     st.markdown("### Final Score")
#                     if percentage >= 80:
#                         st.success(f"{score}/{total} ({percentage:.0f}%)")
#                     elif percentage >= 60:
#                         st.warning(f"{score}/{total} ({percentage:.0f}%)")
#                     else:
#                         st.error(f"{score}/{total} ({percentage:.0f}%)")
        
#         else:  # open_ended
#             st.markdown(f"## {quiz_data['topic']} Quiz")
            
#             # Create form for open-ended questions
#             with st.form(key=f"open_quiz_form_{quiz_data['topic']}"):
#                 answers = {}
#                 for i, question in enumerate(quiz_data['questions']):
#                     st.markdown(f"### Question {i+1}")
#                     st.write(question['question'])
                    
#                     answer = st.text_area(
#                         "Your answer:",
#                         key=f"open_q_{i}",
#                         height=150
#                     )
#                     answers[i] = answer
#                     st.markdown("---")
                
#                 submitted = st.form_submit_button("Submit Quiz")
                
#                 if submitted:
#                     total_score = 0
#                     num_answered = 0
                    
#                     for i, question in enumerate(quiz_data['questions']):
#                         if answers[i].strip():
#                             feedback = st.session_state.quiz_agent.grade_open_ended_response(
#                                 question, 
#                                 answers[i]
#                             )
#                             score = int(feedback['score'])
#                             total_score += score
#                             num_answered += 1
                            
#                             st.markdown(f"### Question {i+1} Feedback")
#                             if score >= 80:
#                                 st.success(f"Score: {score}/100")
#                             elif score >= 60:
#                                 st.warning(f"Score: {score}/100")
#                             else:
#                                 st.error(f"Score: {score}/100")
                            
#                             with st.expander("View detailed feedback"):
#                                 st.write("**Feedback:**", feedback['feedback'])
#                                 st.write("**Key points covered:**")
#                                 for point in feedback['key_points_covered']:
#                                     st.success(f"✓ {point}")
#                                 if feedback['missing_points']:
#                                     st.write("**Missing points:**")
#                                     for point in feedback['missing_points']:
#                                         st.error(f"✗ {point}")
#                                 st.write("**Suggestions for improvement:**")
#                                 for suggestion in feedback['improvement_suggestions']:
#                                     st.info(f"• {suggestion}")
                            
#                             st.markdown("---")
                    
#                     if num_answered > 0:
#                         average_score = total_score / num_answered
#                         st.markdown("### Overall Score")
#                         if average_score >= 80:
#                             st.success(f"Average Score: {average_score:.1f}%")
#                         elif average_score >= 60:
#                             st.warning(f"Average Score: {average_score:.1f}%")
#                         else:
#                             st.error(f"Average Score: {average_score:.1f}%")

# def app():
#     st.title("📚 Study Tools")
    
#     if 'cheatsheet_agent' not in st.session_state or 'quiz_agent' not in st.session_state:
#         st.error("Please initialize the application properly")
#         return

#     # Tool Selection
#     tool_type = st.radio(
#         "Choose a study tool:",
#         ["Cheat Sheet Generator", "Quiz Generator"]
#     )
    
#     topic = st.text_input("Enter the topic:")
    
#     if tool_type == "Cheat Sheet Generator":
#         if st.button("Generate Cheat Sheet"):
#             with st.spinner("Creating your cheat sheet..."):
#                 result = st.session_state.cheatsheet_agent.create_cheatsheet(topic)
#                 if "error" in result:
#                     st.error(result["error"])
#                 else:
#                     st.markdown(result["content"])
#                     with st.expander("Sources"):
#                         st.write(result["sources"])
                    
#     else:  # Quiz Generator
#         col1, col2 = st.columns(2)
#         with col1:
#             num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=5)
#         with col2:
#             question_type = st.selectbox("Question type:", ["multiple_choice", "open_ended"])
            
#         if st.button("Generate Quiz"):
#             if not topic:
#                 st.error("Please enter a topic first!")
#                 return
            
#             with st.spinner("Creating your quiz..."):
#                 try:
#                     result = st.session_state.quiz_agent.generate_quiz(
#                         topic=topic,
#                         num_questions=num_questions,
#                         question_type=question_type
#                     )
                    
#                     if "error" in result:
#                         st.error(result["error"])
#                     else:
#                         display_quiz_in_container(result, question_type)
#                 except Exception as e:
#                     st.error(f"Error generating quiz: {str(e)}")
#                     if st.checkbox("Show error details"):
#                         st.code(str(e))










import streamlit as st
from pages.quiz_component import create_quiz_component

def app():
    st.title("📚 Study Tools")
    
    if 'cheatsheet_agent' not in st.session_state or 'quiz_agent' not in st.session_state:
        st.error("Please initialize the application properly")
        return

    # Tool Selection
    tool_type = st.radio(
        "Choose a study tool:",
        ["Cheat Sheet Generator", "Quiz Generator"],
        key="tool_selector"
    )
    
    topic = st.text_input("Enter the topic:")
    
    if tool_type == "Cheat Sheet Generator":
        if st.button("Generate Cheat Sheet", key="cheatsheet_btn"):
            with st.spinner("Creating your cheat sheet..."):
                result = st.session_state.cheatsheet_agent.create_cheatsheet(topic)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.markdown(result["content"])
                    with st.expander("Sources"):
                        st.write(result["sources"])
                    
    else:  # Quiz Generator
        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.number_input(
                "Number of questions:", 
                min_value=1, 
                max_value=10, 
                value=5,
                key="num_questions"
            )
        with col2:
            question_type = st.selectbox(
                "Question type:", 
                ["multiple_choice", "open_ended"],
                key="question_type"
            )
            
        if st.button("Generate Quiz", key="quiz_btn"):
            if not topic:
                st.error("Please enter a topic first!")
                return
            
            with st.spinner("Creating your quiz..."):
                try:
                    result = st.session_state.quiz_agent.generate_quiz(
                        topic=topic,
                        num_questions=num_questions,
                        question_type=question_type
                    )
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Use the new quiz component
                        create_quiz_component(result, question_type)
                        
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")
                    st.write(f"Debug: {str(e)}")
