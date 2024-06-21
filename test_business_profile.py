from dotenv import load_dotenv
from crewai import Agent, Process, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool

load_dotenv()

business_profile_output_format = """
      Output format guidelines:
      Executive Summary: An overview to summarise key findings, focus on most critical info (name, industry, headquarter location).
      Detailed Business Profile:
      Company Background: Historical info ,mission and founders and key executives.
      Product and Service Analysis: List of products/services, their unique selling points, recent developments.
      Market Position: Industry standing and market share.
      Competitor Analysis: Analyse key competitors, competitive advantages of the company, identify market gaps.
      Customer Segmentation: target market, geographical reach and demographic information.
      Customer Insights: Customer needs and pain points, behavior and feedback.
      Financial Performance: Revenue, profitability and funding/investment.
      Technological Insights: Notable technology usage (hardware and software), technology adoption rate.
      Strategic Insights and Pain Points:
      Recent Developments: Strategic moves (mergers or partnerships).
      Challenges: Operational, market, technological and strategic.
      Conclusion and Recommendations: Conclusions based on data and provide actionable recommendations for strategic planning and sales strategies
      References: List all links you found. For Proff, link this https://apidocs.proff.no/, do not link the direct API request.
      """

serper_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool()

company_research_agent = Agent(
    role="B2B Sales Research Expert",
    goal="Conduct a comprehensive research on a business with information provided",
    backstory="As a senior sales researcher, you excel at using search tools to"
    "find information that could be vital for B2B company to approach the business.",
    verbose=True,
    tools=[serper_tool, website_search_tool],
    max_iter=10,
)

business_profile_agent = Agent(
    role="Expert Business Consultant",
    goal="From the provided data, create a detailed business profile for a business",
    backstory="As a senior business consultant, you can identify information that can"
    "be essential for the process of creating a detailed business profile from the provided data."
    "Remember that the business profile will be used to create an Ideal Customer Profile for the company"
    "for easy company segmentation.",
    verbose=True,
    max_iter=10,
)

manager_agent = Agent(
    role="Manager",
    goal="Ensure the smooth operation and coordination of the team",
    verbose=True,
    backstory=("As a seasoned project manager, you excel in organizing tasks."),
    max_iter=10,
)

research_task = Task(
    description="Using the search tools provided to"
    "conduct an in-depth research on a business. Some information about the business is provided for you"
    "to know where to start {proff_data}. The output must provide an in-depth understanding about the business "
    "information regarding firmographic, demographic, technographic and goals + challenges.",
    expected_output="A list of sources and some significant information you got from the sources.",
    tools=[serper_tool, website_search_tool],
    callback="search_data",
    agent=company_research_agent,
    human_input=True,
)

business_profile_task = Task(
    description="You are provided with the information of a company from an API and from a comprehensive internet search."
    "Your job is to analyse the data an create a detailed business profile.",
    expected_output=business_profile_output_format,
)


crew = Crew(
    agents=[company_research_agent, business_profile_agent],
    tasks=[research_task, business_profile_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=10,
    manager_agent=manager_agent,
)
proff_data = str(
    "{companyType:ENK,companyTypeName:Enkeltpersonforetak,hasSecurity:false,proffListingId:IFELH6Q01DI,registeredAsEnterprise:false,registeredForNav:false,registeredForPayrollTax:false,registeredForVat:false,registeredForVoluntary:false,registrationDate:03.06.2023,shareCapital:0,yearsInOperation:1,contactPerson:{name:Malin Kristin Fonell,role:Innehaver,roleCode:INNH},personRoles:[{birthDate:19011991,name:Malin Kristin Fonell,personId:2429719,title:Innehaver,titleCode:INNH,postalAddress:{addressLine:Marcus Rone Toftes Gate 59 A,postPlace:OSLO,zipCode:0552},details:{href:https://api.proff.no/persons/business/NO/2429719,rel:related}}],mortgages:{overview:{voluntary:{count:0,sum:0},compulsory:{count:0,sum:0},other:{count:0,sum:0}}},centralApproval:{status:{approved:false}},companyId:931509314,establishedDate:2023-06-03,foundationYear:2023,marketingProtection:false,naceCategories:[73.110 Reklamebyråer],name:+ FONELL,organisationNumber:931509314,link:{href:https://api.proff.no/companies/register/NO/931509314,rel:self},location:{countryPart:Østlandet,county:Oslo,municipality:Oslo,coordinates:[{XCoordinate:10.76088245,YCoordinate:59.92279314,coordinateSystem:EPSG:4326}]},phoneNumbers:{},postalAddress:{addressLine:c/o Marcus Rone Toftes gate 59A,postPlace:Oslo,zipCode:0552},status:{statusFlag:ACTIVE},visitorAddress:{addressLine:c/o Marcus Rone Toftes gate 59A,postPlace:Oslo,zipCode:0552},sectorCode:8200}"
)

result = crew.kickoff()
print(result)
print(crew.usage_metrics)
