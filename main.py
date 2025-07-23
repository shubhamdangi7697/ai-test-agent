import streamlit as st
import google.generativeai as genai
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import random

# Configure Streamlit page
st.set_page_config(
    page_title="AI Practice Test Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PracticeTestGenerator:
    def __init__(self, api_key: str):
        """Initialize the practice test generator with Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Define practice domains
        self.domains = {
            "aws_cloud": {
                "name": "AWS Cloud Practitioner",
                "icon": "☁️",
                "description": "AWS Cloud services, architecture, pricing, and security",
                "topics": ["EC2", "S3", "VPC", "IAM", "RDS", "Lambda", "CloudFormation", "Security", "Pricing", "Well-Architected"]
            },
            "gen_ai": {
                "name": "Generative AI Expert",
                "icon": "🤖",
                "description": "Machine Learning, Deep Learning, LLMs, and AI applications",
                "topics": ["Neural Networks", "Transformers", "GPT", "Computer Vision", "NLP", "MLOps", "Ethics", "Fine-tuning", "RAG", "Prompt Engineering"]
            },
            "software_engineer": {
                "name": "Software Engineer",
                "icon": "💻",
                "description": "Programming, algorithms, system design, and software development",
                "topics": ["Data Structures", "Algorithms", "System Design", "Database Design", "OOP", "Design Patterns", "Testing", "API Design", "Microservices", "Performance"]
            },
            "scrum_master": {
                "name": "Scrum Master",
                "icon": "📋",
                "description": "Agile methodologies, Scrum framework, and team management",
                "topics": ["Scrum Framework", "Sprint Planning", "Daily Standups", "Retrospectives", "User Stories", "Backlog Management", "Team Dynamics", "Agile Principles", "Facilitation", "Coaching"]
            },
            "data_scientist": {
                "name": "Data Scientist",
                "icon": "📊",
                "description": "Statistics, ML algorithms, data analysis, and visualization",
                "topics": ["Statistics", "Machine Learning", "Python/R", "Data Visualization", "Feature Engineering", "Model Evaluation", "Big Data", "A/B Testing", "SQL", "Business Intelligence"]
            },
            "devops_engineer": {
                "name": "DevOps Engineer",
                "icon": "🔧",
                "description": "CI/CD, containerization, infrastructure, and automation",
                "topics": ["Docker", "Kubernetes", "CI/CD", "Infrastructure as Code", "Monitoring", "Linux", "Networking", "Security", "Automation", "Cloud Platforms"]
            },
            "product_manager": {
                "name": "Product Manager",
                "icon": "🎯",
                "description": "Product strategy, roadmapping, and stakeholder management",
                "topics": ["Product Strategy", "Market Research", "User Research", "Product Roadmap", "Metrics", "Stakeholder Management", "Pricing", "Go-to-Market", "Feature Prioritization", "Analytics"]
            },
            "cybersecurity": {
                "name": "Cybersecurity Specialist",
                "icon": "🔒",
                "description": "Security frameworks, threat analysis, and risk management",
                "topics": ["Network Security", "Cryptography", "Risk Assessment", "Incident Response", "Compliance", "Penetration Testing", "Security Architecture", "Identity Management", "Threat Intelligence", "Security Monitoring"]
            }
        }
    
    def generate_questions_prompt(self, domain: str, question_count: int = 50) -> str:
        """Generate prompt for creating practice questions"""
        domain_info = self.domains[domain]
        
        prompt = f"""
You are an expert in {domain_info['name']} creating a comprehensive practice test.

Generate exactly {question_count} high-quality practice questions for {domain_info['name']}.

REQUIREMENTS:
- Cover all these topics evenly: {', '.join(domain_info['topics'])}
- Difficulty distribution: 20% easy, 60% medium, 20% hard
- Include scenario-based and practical questions
- Each question must have exactly 4 multiple choice options (A, B, C, D)
- Questions should be realistic and industry-relevant
- Provide detailed explanations for correct answers

OUTPUT FORMAT - Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": 1,
      "question": "What is the primary benefit of using AWS Lambda for microservices architecture?",
      "options": {{
        "A": "Lower cost than EC2 instances",
        "B": "Automatic scaling and serverless execution",
        "C": "Better security than container services",
        "D": "Unlimited execution time"
      }},
      "correct_answer": "B",
      "explanation": "AWS Lambda automatically scales based on demand and follows a serverless model where you pay only for compute time consumed. This makes it ideal for microservices as it eliminates server management overhead.",
      "topic": "Lambda",
      "difficulty": "medium"
    }}
  ]
}}

IMPORTANT: 
- Generate exactly {question_count} questions
- Ensure JSON is properly formatted
- Make questions challenging but fair
- Include current best practices and latest updates
- Focus on real-world scenarios and practical knowledge

Generate the questions now:
"""
        return prompt
    
    def generate_questions(self, domain: str, question_count: int = 50) -> Dict:
        """Generate practice questions using Gemini"""
        try:
            prompt = self.generate_questions_prompt(domain, question_count)
            
            # Configure generation parameters for better JSON output
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json"
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse the JSON response
            questions_data = json.loads(response.text)
            
            # Validate and clean the data
            if "questions" not in questions_data:
                raise ValueError("Invalid response format: missing 'questions' key")
            
            # Ensure we have the right number of questions
            questions = questions_data["questions"]
            if len(questions) < question_count * 0.8:  # Allow some tolerance
                st.warning(f"Generated {len(questions)} questions instead of {question_count}")
            
            # Add metadata
            questions_data["metadata"] = {
                "domain": domain,
                "generated_at": datetime.now().isoformat(),
                "total_questions": len(questions),
                "model": "gemini-2.0-flash-exp"
            }
            
            return questions_data
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response: {str(e)}", "raw_response": response.text if 'response' in locals() else "No response"}
        except Exception as e:
            return {"error": f"Failed to generate questions: {str(e)}"}
    
    def generate_questions_batch(self, domain: str, total_questions: int = 50) -> Dict:
        """Generate questions in batches for better reliability"""
        batch_size = 25  # Generate in smaller batches
        all_questions = []
        
        try:
            batches_needed = (total_questions + batch_size - 1) // batch_size
            
            for batch_num in range(batches_needed):
                questions_in_batch = min(batch_size, total_questions - len(all_questions))
                
                st.info(f"Generating batch {batch_num + 1}/{batches_needed} ({questions_in_batch} questions)...")
                
                batch_result = self.generate_questions(domain, questions_in_batch)
                
                if "error" in batch_result:
                    return batch_result
                
                all_questions.extend(batch_result["questions"])
                
                # Small delay between batches
                time.sleep(1)
            
            return {
                "questions": all_questions,
                "metadata": {
                    "domain": domain,
                    "generated_at": datetime.now().isoformat(),
                    "total_questions": len(all_questions),
                    "model": "gemini-2.0-flash-exp",
                    "batches_used": batches_needed
                }
            }
            
        except Exception as e:
            return {"error": f"Batch generation failed: {str(e)}"}

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'current_domain' not in st.session_state:
        st.session_state.current_domain = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'test_started' not in st.session_state:
        st.session_state.test_started = False
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None

def reset_test():
    """Reset test session"""
    st.session_state.current_question = 0
    st.session_state.user_answers = {}
    st.session_state.test_started = False
    st.session_state.test_completed = False
    st.session_state.start_time = None
    st.session_state.questions = []

def calculate_results():
    """Calculate test results"""
    if not st.session_state.questions:
        return {}
    
    total_questions = len(st.session_state.questions)
    correct_answers = 0
    topic_performance = {}
    difficulty_performance = {"easy": {"correct": 0, "total": 0}, 
                            "medium": {"correct": 0, "total": 0}, 
                            "hard": {"correct": 0, "total": 0}}
    
    for i, question in enumerate(st.session_state.questions):
        user_answer = st.session_state.user_answers.get(i)
        correct_answer = question["correct_answer"]
        topic = question.get("topic", "General")
        difficulty = question.get("difficulty", "medium")
        
        # Track topic performance
        if topic not in topic_performance:
            topic_performance[topic] = {"correct": 0, "total": 0}
        topic_performance[topic]["total"] += 1
        
        # Track difficulty performance
        if difficulty in difficulty_performance:
            difficulty_performance[difficulty]["total"] += 1
        
        if user_answer == correct_answer:
            correct_answers += 1
            topic_performance[topic]["correct"] += 1
            if difficulty in difficulty_performance:
                difficulty_performance[difficulty]["correct"] += 1
    
    score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    return {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "score_percentage": score_percentage,
        "topic_performance": topic_performance,
        "difficulty_performance": difficulty_performance
    }

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Initialize the generator with Gemini API key
    if 'generator' not in st.session_state:
        api_key = st.secrets.get("GEMINI_API_KEY", "AIzaSyBYxBSfyw6XGt2EPvRDwijQwqpklnY5AfY")  # Add your Gemini API key
        if api_key == "your-gemini-api-key-here":
            st.error("Please add your Gemini API key to .streamlit/secrets.toml")
            st.code("""
# Add this to .streamlit/secrets.toml:
GEMINI_API_KEY = "your-actual-gemini-api-key"
            """)
            st.stop()
        st.session_state.generator = PracticeTestGenerator(api_key)
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 AI Practice Test Generator")
        st.markdown("*Powered by Gemini 2.0 Flash*")
        st.markdown("---")
        
        if st.session_state.test_started:
            st.metric("Progress", f"{st.session_state.current_question + 1}/{len(st.session_state.questions)}")
            progress = (st.session_state.current_question + 1) / len(st.session_state.questions)
            st.progress(progress)
            
            if st.button("🔄 Reset Test", type="secondary"):
                reset_test()
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Available Domains")
        for domain_key, domain_info in st.session_state.generator.domains.items():
            st.markdown(f"**{domain_info['icon']} {domain_info['name']}**")
            st.caption(domain_info['description'])
    
    # Main content
    if not st.session_state.test_started:
        # Domain selection page
        st.title("🎯 AI-Powered Practice Test Generator")
        st.markdown("### Choose your practice domain and start testing your knowledge!")
        st.markdown("*Powered by Google Gemini 2.0 Flash for fresh, AI-generated questions*")
        
        # Display domain cards
        cols = st.columns(2)
        domain_keys = list(st.session_state.generator.domains.keys())
        
        for i, domain_key in enumerate(domain_keys):
            domain_info = st.session_state.generator.domains[domain_key]
            
            with cols[i % 2]:
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 20px; border: 2px solid #f0f2f6; border-radius: 10px; margin: 10px 0; background-color: #fafafa;">
                        <h3>{domain_info['icon']} {domain_info['name']}</h3>
                        <p>{domain_info['description']}</p>
                        <p><strong>Topics:</strong> {', '.join(domain_info['topics'][:5])}...</p>
                        <p><small>🎯 50+ AI-generated questions</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Start {domain_info['name']} Test", key=f"start_{domain_key}", type="primary"):
                        st.session_state.current_domain = domain_key
                        
                        with st.spinner(f"🤖 Gemini is generating 50 questions for {domain_info['name']}..."):
                            # Use batch generation for better reliability
                            questions_data = st.session_state.generator.generate_questions_batch(domain_key, 50)
                            
                            if "error" in questions_data:
                                st.error(f"❌ Error: {questions_data['error']}")
                                if "raw_response" in questions_data:
                                    with st.expander("Debug Info"):
                                        st.text(questions_data['raw_response'][:1000])
                            else:
                                st.session_state.questions = questions_data["questions"]
                                st.session_state.test_started = True
                                st.session_state.start_time = datetime.now()
                                st.success(f"✅ Generated {len(st.session_state.questions)} questions!")
                                time.sleep(1)
                                st.rerun()
    
    elif st.session_state.test_started and not st.session_state.test_completed:
        # Question display page
        if st.session_state.questions:
            current_q = st.session_state.current_question
            question = st.session_state.questions[current_q]
            domain_info = st.session_state.generator.domains[st.session_state.current_domain]
            
            # Header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.title(f"{domain_info['icon']} {domain_info['name']} Test")
            with col2:
                st.metric("Question", f"{current_q + 1}/{len(st.session_state.questions)}")
            with col3:
                difficulty_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                difficulty = question.get('difficulty', 'medium')
                st.metric("Difficulty", f"{difficulty_color.get(difficulty, '🟡')} {difficulty.title()}")
            
            st.markdown("---")
            
            # Question
            st.markdown(f"### Question {current_q + 1}")
            st.markdown(f"**Topic:** {question.get('topic', 'General')}")
            st.markdown(f"#### {question['question']}")
            
            # Options
            user_answer = st.radio(
                "Choose your answer:",
                options=list(question['options'].keys()),
                format_func=lambda x: f"{x}) {question['options'][x]}",
                key=f"q_{current_q}",
                index=None
            )
            
            # Store answer
            if user_answer:
                st.session_state.user_answers[current_q] = user_answer
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("⏭️ Skip Question"):
                    if current_q < len(st.session_state.questions) - 1:
                        st.session_state.current_question += 1
                        st.rerun()
                    else:
                        st.session_state.test_completed = True
                        st.rerun()
            
            with col2:
                if st.button("✅ Submit Answer", disabled=not user_answer):
                    if current_q < len(st.session_state.questions) - 1:
                        st.session_state.current_question += 1
                        st.rerun()
                    else:
                        st.session_state.test_completed = True
                        st.rerun()
            
            with col3:
                if st.button("🏁 Finish Test"):
                    st.session_state.test_completed = True
                    st.rerun()
            
            # Progress indicator
            st.markdown("---")
            progress = (current_q + 1) / len(st.session_state.questions)
            st.progress(progress)
            st.caption(f"Progress: {current_q + 1}/{len(st.session_state.questions)} questions")
    
    elif st.session_state.test_completed:
        # Results page
        results = calculate_results()
        domain_info = st.session_state.generator.domains[st.session_state.current_domain]
        
        st.title(f"🎉 Test Results - {domain_info['name']}")
        
        # Overall score
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score_color = "🟢" if results['score_percentage'] >= 80 else "🟡" if results['score_percentage'] >= 60 else "🔴"
            st.metric("Score", f"{score_color} {results['score_percentage']:.1f}%")
        with col2:
            st.metric("Correct", f"{results['correct_answers']}/{results['total_questions']}")
        with col3:
            if st.session_state.start_time:
                duration = datetime.now() - st.session_state.start_time
                minutes = duration.seconds // 60
                seconds = duration.seconds % 60
                st.metric("Duration", f"{minutes}:{seconds:02d}")
        with col4:
            grade = "A" if results['score_percentage'] >= 90 else "B" if results['score_percentage'] >= 80 else "C" if results['score_percentage'] >= 70 else "D" if results['score_percentage'] >= 60 else "F"
            grade_color = "🟢" if grade in ["A", "B"] else "🟡" if grade == "C" else "🔴"
            st.metric("Grade", f"{grade_color} {grade}")
        
        # Performance by topic
        st.markdown("### 📊 Performance by Topic")
        if results['topic_performance']:
            topic_data = []
            for topic, performance in results['topic_performance'].items():
                percentage = (performance['correct'] / performance['total']) * 100 if performance['total'] > 0 else 0
                topic_data.append({
                    "Topic": topic,
                    "Correct": performance['correct'],
                    "Total": performance['total'],
                    "Percentage": f"{percentage:.1f}%"
                })
            
            st.dataframe(topic_data, use_container_width=True)
        
        # Performance by difficulty
        st.markdown("### ⚡ Performance by Difficulty")
        diff_cols = st.columns(3)
        difficulty_colors = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
        
        for i, (difficulty, performance) in enumerate(results['difficulty_performance'].items()):
            if performance['total'] > 0:
                percentage = (performance['correct'] / performance['total']) * 100
                with diff_cols[i]:
                    color = difficulty_colors.get(difficulty, "⚪")
                    st.metric(
                        f"{color} {difficulty.title()}", 
                        f"{percentage:.1f}%",
                        f"{performance['correct']}/{performance['total']}"
                    )
        
        # Question review
        st.markdown("### 📝 Question Review")
        with st.expander("Review All Questions"):
            for i, question in enumerate(st.session_state.questions):
                user_answer = st.session_state.user_answers.get(i, "Not answered")
                correct_answer = question["correct_answer"]
                is_correct = user_answer == correct_answer
                
                icon = "✅" if is_correct else "❌" if user_answer != "Not answered" else "⏭️"
                st.markdown(f"**{icon} Question {i+1}:** {question['question']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    answer_color = "🟢" if is_correct else "🔴" if user_answer != "Not answered" else "⚪"
                    st.markdown(f"**Your answer:** {answer_color} {user_answer}")
                with col2:
                    st.markdown(f"**Correct answer:** 🟢 {correct_answer}")
                
                if not is_correct and user_answer != "Not answered":
                    st.markdown(f"**💡 Explanation:** {question['explanation']}")
                
                st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Take Another Test", type="primary"):
                reset_test()
                st.rerun()
        with col2:
            # Create downloadable results
            results_json = {
                "domain": st.session_state.current_domain,
                "results": results,
                "timestamp": datetime.now().isoformat(),
                "model": "gemini-2.0-flash-exp"
            }
            st.download_button(
                "📊 Download Results",
                json.dumps(results_json, indent=2),
                f"test_results_{st.session_state.current_domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
