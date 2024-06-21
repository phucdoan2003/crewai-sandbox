from crewai import Agent
from crewai_tools import SerperDevTool
from crewai import Task
from crewai import Crew, Process
from dotenv import load_dotenv

load_dotenv()

search_tool = SerperDevTool()

researcher = Agent(
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."
    ),
    tools=[search_tool],
    max_iter=5,
)

writer = Agent(
    role="Writer",
    goal="Narrate compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner"
    ),
    allow_delegation=False,
    max_iter=5,
)

manager = Agent(
    role="manager",
    goal="Ensure the smooth operation and coordination of the team",
    verbose=True,
    backstory=(
        "As a seasoned project manager, you excel in organizing"
        "tasks, managing timelines, and ensuring the team stays on track"
    ),
)

research_task = Task(
    description=(
        "Identify the next big trend in {topic}."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points,"
        "its market opportunites, and potential risks."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    tools=[search_tool],
    agent=researcher,
    callback="research_callback",
    human_input=True,
)

write_task = Task(
    description=(
        "Compose an insightful article on {topic}."
        "Focus on the latest trends and how it's impacting the industry."
        "This article should be easy to understand, engaging, and positive."
    ),
    expected_output="A 4 paragraph article on {topic} advancements formatted as markdown.",
    agent=writer,
    output_file="new-blog-post.md",
)


crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    manager_agent=manager,
)

result = crew.kickoff(inputs={"topic": "AI in healthcare"})
print(result)
print(crew.usage_metrics)
