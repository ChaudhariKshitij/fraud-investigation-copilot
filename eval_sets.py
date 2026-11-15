"""Retrieval evaluation sets.

DEV: the 10 queries used while the first two failures were found and diagnosed.
HELDOUT: 20 new queries written BEFORE the retriever was changed, never used for tuning.
Only the held-out numbers should be treated as an honest estimate.
"""
V, A, G, S = ("Card Velocity Attack", "Sudden Amount Spike", "Unusual Geography At Night", "Structuring Below Threshold")
R, F = "Investigation Report Standards", "False Positive Guidance"

DEV = [
    ("many small purchases on one card within a few minutes", V),
    ("card testing burst of online transactions", V),
    ("one huge purchase far above normal spending", A),
    ("customer bought expensive electronics, 20 times usual", A),
    ("transactions from a foreign country at 2am", G),
    ("card used abroad overnight, customer never travels", G),
    ("payments just under the 10000 reporting limit", S),
    ("how do I spot structuring", S),
    ("what must a case report contain", R),
    ("what could explain a large legitimate purchase", F),
]

HELDOUT = [
    ("a dozen online payments in twenty minutes from a new device", V),
    ("stolen card numbers being tested with rapid low-value charges", V),
    ("what should I do when a card shows rapid repeated charges", V),
    ("one transaction is enormous compared with the customer's median", A),
    ("sudden big electronics order after months of small spends", A),
    ("how to handle an unusually large payment on an account", A),
    ("the cardholder is in India but a swipe happened in the US in the early morning", G),
    ("two purchases abroad one after another during the night", G),
    ("should I ask the customer if they are travelling", G),
    ("several transfers of 9,500 over a day", S),
    ("someone splitting money into amounts that avoid review", S),
    ("escalate to compliance for suspicious activity review", S),
    ("what evidence must be written in a fraud case file", R),
    ("how should I word a conclusion without claiming certainty", R),
    ("what should a report say it could not verify", R),
    ("could a salary credit explain a big purchase", F),
    ("customer on holiday, is this really fraud", F),
    ("subscription renewal or bill batch that looks odd", F),
    ("check prior history before closing as fraud", F),
    ("identify bursts of tiny charges then cash-out", V),
]

# FRESH: 20 more queries written AFTER the retriever fix, run once, never tuned on.
# This is the only set that is an unbiased estimate of the improved retriever.
FRESH = [
    ("fifteen quick purchases from an unfamiliar device in under half an hour", V),
    ("what steps do I take to stop a card that is being drained", V),
    ("customer's spending jumped to many times their normal level in one go", A),
    ("a single order worth far more than anything in this account's past", A),
    ("how do I verify a high value purchase before approving it", A),
    ("login and swipe from a country the customer has no history in, around 3 in the morning", G),
    ("cloned card suspected because use appeared overseas while owner was at home", G),
    ("repeated payments of 9,800 spread over a weekend", S),
    ("amounts clustered right below the point that triggers automatic checks", S),
    ("do we tell the customer when we suspect they are breaking up payments", S),
    ("list of things every case write-up has to include", R),
    ("is it okay to call a customer a fraudster in the report", R),
    ("what limitations should be mentioned in an automated assessment", R),
    ("which innocent behaviours commonly trigger alerts", F),
    ("customer just got paid and bought a television", F),
    ("recurring payment that looks like a pattern but is not suspicious", F),
    ("how many charges per hour is suspicious for one card", V),
    ("spending ten times above average on gadgets", A),
    ("night time card usage in another country", G),
    ("payments kept just beneath the 10,000 mark", S),
]
