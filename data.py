import pandas as pd
import csv

data = [
    {
        "article": "Last year, Paypal closed accounts run by Toby Young, who is general secretary of the Free Speech Union. They were later reinstated by the US payments company. The government subsequently announced a review into payment services regulations, including the practice of firms apparently closing down the accounts of people or businesses that hold views the lender does not agree with.",
        "category": "business",
    },
    {
        "article": "UK Finance, which represents the banking industry, said lenders should discuss the closure of an account with a customer so far as is feasible and permissible.It said though there will be situations where it may not be appropriate or permissible for a bank to engage in a dialogue to explain their reasoning.This would include a breach of terms and conditions, abusive or threatening behaviour to colleagues or if banks have been directed not to by regulators, HM Government, police and other authorities. Mr Farage said he approached seven other banks to open personal and business accounts and was turned down by all of them.",
        "category": "business",
    },
    {
        "article": "Electric carmaker Tesla says it delivered a record number of vehicles in the three months to the end of June, after cutting prices to boost sales. It has lowered prices in markets including the US, UK and China to compete with rival manufacturers. This weekend, major Chinese car makers also reported a surge in sales in June.",
        "category": "business",
    },
    {
        "article": "As recently as 9 May, a vast tanker capable of carrying more than 160,000 cubic metres of gas compressed into liquid form - liquefied natural gas or LNG - pulled out of the port of Sabetta, on the Yamal peninsula in Russia's far north. That cargo was purchased by Shell before heading onwards to its ultimate destination, Hong Kong. It is one of eight LNG cargoes that Shell has bought from Yamal this year, according to data from the Kpler database analysed by Global Witness. Last year Shell accounted for 12 percentage of Russia's seaborne LNG trade, Global Witness calculates, and was among the top five traders of Russian-originated LNG that year.",
        "category": "business",
    },
    {
        "article": "Last year, the UK government introduced a windfall tax on profits made from extracting UK oil and gas - called the Energy Profits Levy (EPL) - to help fund its scheme to lower gas and electricity bills. Under the government's Energy Price Guarantee, energy bills for a typical household have been limited to £2,500 a year, although this level of support is due to stop at the end of June.",
        "category": "business",
    },
    {
        "article": "In February, Shell reported profits of $39.9bn for 2022, double the previous year's total and the highest in its 115-year history. While the jump in oil and gas prices following the start of the war in Ukraine led to big profits for energy companies, it also fuelled a rise in energy bills for households and businesses.",
        "category": "business",
    },
    {
        "article": "Switching bank holidays helps me in saving money we used to spend on childcare. If we have nothing planned during bank holidays, then my husband takes care of the kids, then I can use those extra days during summer holidays. It's an extra eight days so we don't have to worry about childcare for almost a week and a half.",
        "category": "business",
    },
    {
        "article": "Manchester-based crisis charity Human Appeal allowed its 170-strong workforce to swap this year's Good Friday and Easter Monday bank holidays to extend their Eid celebrations, a no-brainer for the 70% who identify as Muslim. The more appealing option was a full week off at the conclusion of Ramadan in May, made up of the early May bank holiday, two holiday days given by Human Appeal to celebrate Eid, plus the two days swapped over from Easter.",
        "category": "business",
    },
    {
        "article": "With energy and food prices now soaring, firms in those sectors face very specific pricing and supply issues that they need to be able to deal with. However, Dr Pescaroli says all companies should prepare a list of practical things to check and tick off in the event of a crisis, and more importantly - before one.",
        "category": "business",
    },
    {
        "article": "Firms also need to have back-up generators to supply power if networks go down. And they need to train more than one member of staff to be able to run them - in case he or she is away on leave, if the mains power supply is cut. Large companies that are thinking ahead are now appointing risk managers at board level. This person will be someone whose sole responsibility is to ensure the firm can survive the next major crisis. They are preparing for the worst, and even gaming different scenarios to see how the company would respond.",
        "category": "business",
    },
    {
        "article": "Russia's seventh seed Andrey Rublev is a potential opponent for Djokovic in the quarter-finals and he began his campaign with a 6-3 7-5 6-4 success over Australia's Max Purcell. Rublev, unable to play at Wimbledon in 2022 because of the ban handed out to Russian and Belarusian players after the Russian invasion of Ukraine, was only in trouble in the second set at 5-2 down before he won five games in a row in a 6-3 7-5 6-4 victory.",
        "category": "sport",
    },
    {
        "article": "Norwegian fourth seed Casper Ruud, who lost to Djokovic in the recent French Open final, was tested by Laurent Lokoli of France before eventually going through in four sets on Court One. Elsewhere, there were successes for 14th seed Lorenzo Musetti and 17th seed Hubert Hurkacz against Juan Pablo Varillas of Peru and Albert Ramos-Vinolas of Spain respectively.",
        "category": "sport",
    },
    {
        "article": "New owner Todd Boehly sacked Thomas Tuchel and Graham Potter in his first season, with the club spending about £288m in January, which was more than the combined total of all clubs in the Bundesliga, La Liga, Serie A and Ligue 1. Pochettino has taken over a squad that has seen more change this summer, with Kalidou Koulibaly,Edouard Mendyand N'Golo Kanteall leaving to join teams in Saudi Arabia, England midfielderMason Mount agreeing to join Manchester United, Ruben Loftus-Cheekmoving to AC Milan and Mateo Kovacic signing for champions Manchester City.",
        "category": "sport",
    },
    {
        "article": "We are here to try to help the club and the fans. In the end, the most important thing in football is for them to be happy and to feel proud of us and in the way we approach games. The players need to know that.",
        "category": "sport",
    },
    {
        "article": "Djokovic, who has not lost on Centre Court since being beaten by Andy Murray in the 2013 final, was a break up in the first set, but complained the surface was getting slippery with drops of rain falling. He wrapped up the set just before a shower at 14:20 BST, with the roof then fully closed 15 minutes later. But in that time enough water had got on to the court before the covers came on to cause a lengthy delay.",
        "category": "sport",
    },
    {
        "article": "England head coach Brendon McCullum has warned Australia that his side have been galvanised by Jonny Bairstow's controversial dismissal on day five of the second Ashes Test at Lord's. Believing the ball to be dead and the over to have concluded, Bairstow left his ground as wicketkeeper Alex Carey threw at the stumps and was given out. Australia went on to win by 43 runs and now lead the five-Test series 2-0.",
        "category": "sport",
    },
    {
        "article": "With protesters briefly disrupting the first morning of the second Ashes Test and the Marylebone Cricket Club suspending three members over altercations with Australia players on day five, Yorkshire say they will increase security measures for the third test. A spokesperson for the club said: The wellbeing of players, officials and spectators is paramount, and we are implementing appropriate measures to do everything within our control to keep everyone safe.",
        "category": "sport",
    },
    {
        "article": "A biennial competition featuring all 10 teams from the Six Nations and Rugby Championship is set to start in 2026. It will be played in the existing July and November Test windows in alternate years, outside of British and Irish Lions tours and the World Cup. Two more unions will be invited to join, with plans to introduce promotion and relegation from 2030 via a second-tier competition run by World Rugby.",
        "category": "sport",
    },
    {
        "article": "Establishing the two competitions will pave the way for promotion and relegation matches, contributing towards a valuable pathway for teams, and will support ambitions to sustain and grow the global game.",
        "category": "sport",
    },
    {
        "article": "Premier League legend Petr Cech has joined Oxford City Stars as their new goaltender for the 2023-24 season. The former Chelsea and Arsenal goalkeeper grew up playing junior ice hockey in the Czech Republic before taking up football. Following his retirement he has played for Guildford Phoenix and Chelmsford Chieftains, with the Stars playing in the third tier of British ice hockey.",
        "category": "sport",
    },
    {
        "article": "Florida and Texas are seeing some locally acquired cases of malaria - the first spread of the mosquito-transmitted disease inside the US in 20 years, officials warn in a health alert. Active surveillance for more cases is continuing, the Centres for Disease Control says. The risk of catching malaria in the US remains extremely low, it says.",
        "category": "health",
    },
    {
        "article": "Malaria is caused by being bitten by an infected mosquito. People cannot catch it from each other. But the insects catch it from infected people - and the cycle continues. It is common in large areas of Africa, Asia and Central and South America but not the US.",
        "category": "health",
    },
    {
        "article": "There has been an unusual rise in the number of children and teenagers around the world diagnosed with type 1 diabetes since Covid, say researchers. A new study in JAMA Network Open journal has collated available data from different countries, including the UK, on more than 38,000 young people diagnosed during the pandemic.",
        "category": "health",
    },
    {
        "article": "Experts say it is unclear what has triggered the surge in cases, but there are some theories. One such theory, is that Covid can trigger a reaction in some children which increases the risk of diabetes. But among the studies looking for this type of autoimmune reaction - where the body starts to attack some of its own healthy cells - not all have found evidence to support this theory.",
        "category": "health",
    },
    {
        "article": "Another hypothesis is that exposure to some germs in childhood can help guard against a number of conditions, including diabetes. Some scientists believe it is possible that lockdowns and physical distancing during Covid meant many children did not get sufficient exposure to germs and missed out on this additional protection.",
        "category": "health",
    },
    {
        "article": "Future studies that examine longer-term trends will be important to disentangle the impact of the pandemic from natural fluctuations in incidence of type 1 over time, as well as establishing the range of factors that could be behind any apparent rise.",
        "category": "health",
    },
    {
        "article": "You will find it on the ingredients list of many diet or sugar-free foods including diet drinks, chewing gums and some yoghurts. High profile drinks containing aspartame include Diet Coke, Coke Zero, Pepsi Max, and 7 Up Free, but the sweetener is in around 6,000 food products. The sweetener has been used for decades and approved by food safety bodies, but there has been a swirl of controversy around the ingredient.",
        "category": "health",
    },
    {
        "article": "We do not have the equivalent numbers for aspartame, however, the Joint World Health Organization and Food and Agriculture Organization's Expert Committee on Food Additives is due to report in July. Its stance since 1981 has been a daily intake of 40 milligrams, per kilogram of your body weight, per day was safe. That works out at between 12 and 36 cans of diet drinks (depending on the exact ingredients) a day for a 60 kg (nine-and-a-half stones) adult.",
        "category": "health",
    },
    {
        "article": "There is a constant chance of bird flus making the jump into people. There is a variety of flu viruses in wild birds, and poultry poses a high risk because of the sheer number of farmed animals and their close proximity to people. The 1918 flu pandemic is thought to have started in birds and is estimated to have killed 50 million people. The researchers showed a form of bird flu called H7N9 developed higher levels of resistance to BTN3A3 in 2011 and 2012 before the first human cases emerged in 2013.",
        "category": "health",
    },
    {
        "article": "An appeal for a million pounds has been launched to pay for leukaemia treatment for a 20-month-old girl. Hallie, from Coventry, was diagnosed at Birmingham Children's Hospital when she was eight months old. But when her last stem cell transplant failed to work, her family said they were told their best option might be to pay for treatment in the United States.",
        "category": "health",
    },
]

filename = "articles.csv"
# Extract field names from the first dictionary in the list
fieldnames = data[0].keys()

# Open the CSV file in write mode
with open(filename, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Write the data rows
    writer.writerows(data)
