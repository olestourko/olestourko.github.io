---
layout: post
title: "Compensation package valuation" 
categories: [salary, valuation, investing]
image: 'assets/posts/2018-03-17/banknotes.jpg'
---

This is an article about financially valuing compensation packages, or at least the way I do it. Whether you are
deciding to take an offer or are already employed, its important to be able to financially value the deal.  

## Valuing base salary

Valuing base salary is very simple - the higher the number the better. 

You should compare the salary to the market average for the same role and experience level.
You can use the [StackOverflow Salary Survey](https://insights.stackoverflow.com/survey/2017)
as a rough guide, and find additional market data sources for your region.

Do your due diligence!

## Valuing non-salary compensation

This is where all the trickier valuation questions are. Non-salary compensation is usually a significant part of the package as a developer.
These are things like bonuses, retirement savings plan matching, expense reimbursements, etc...

I think its useful to split these items into three categories based on tangibility:

### Highly Tangible Perks
Things like bonuses, RRSP contribution matching, company stock, etc... are basically deferred cashflows. For example, you have to wait 1 year
to receive a bonus. You have to wait 2 years (or more) for employer RRSP/401k contributions or stock to fully vest. 

- Bonuses
- RRSP / 401k contribution matching
- Company stock

I take the face value of these and [discount](https://en.wikipedia.org/wiki/Discounted_cash_flow)
them for the risk that comes with waiting to recieve them. This ties into the [Time Value of Money](https://www.investopedia.com/terms/t/timevalueofmoney.asp);
 the idea that having a dollar today is worth more than having the same dollar tomorrow.

**Why is this so?**  
Well there's a lot that can happen over that year you wait before receiving an annual bonus:
- Getting laid off that year and not receiving it
- Having to leave because of some family / life event 
- Coming across a better opportunity and having to leave before receiving your bonus
- Missing out on a year of investment gains you would otherwise have had if you invested it
- The bonus being worth slightly less to you a year from now due to price inflation

These are various types of _risks_, and they apply to all deferred cashflow situations.

**So how do you discount for these risks?**  
I use something called the [Net Present Value](https://www.investopedia.com/terms/n/npv.asp) formula, or _NPV_.

$$NPV = \sum_{t=1}^{T}\frac{C_t}{(1 + r)^t}$$

$r$ is the _Discount Rate_  
$C_t$ is the cashflow for time period $t$

**What's the discount rate?**  
Its an interest rate you decide on to account for the risks listed above.
For example):
- 1/5 (20%) chance of getting laid off or taking a new opportunity every year
- 5% gains in the market you miss out on if you had the money to invest now
- 2% annual price inflation

_Discount Rate_ = $(20\% + 5\% + 2\%)= 27\% = 0.27$

Now we can get the risk-adjusted present value for the $5,000 to be paid 1 year from now!

$$ NPV = \frac{\$5,000}{(1 + 0.27) ^ 1}  = \$3,937 $$

_**Note**: we don't do any summation, since there's only 1 time period._

**Let's value something a little more complicated: employer RRSP contributions.**  
Here's an example situation:
- You're a new employee, so you start at year 0 in the employer RRSP program.
- The company does some % match on your personal contributions, up to 5,000$ annually.
- You will be contributing enough to get the full 5,000$ match.
- Employer contributions vest after 2 years. 

We'll use the same discount rate we did to value the bonus; $r = 0.27$.  
We'll consider a cashflow to be an employer contribution that has vested, and we'll compute NPV over 3 years (since it takes
that long to see the first employer contribution vest)  

$$r = 0.27$$  
$$C_1 = \$0 $$  
$$C_2 = \$0 $$  
$$C_3 = \$5,000 $$  

$$ NPV = \frac{\$0}{(1 + 0.27) ^ 1} + \frac{\$0}{(1 + 0.27) ^ 2} + \frac{\$5,000}{(1 + 0.27) ^ 3} = \$2,441 $$


**Notes**  
_1. Alright, so this can be computed more efficiently with the [Present Value](http://financeformulas.net/Present_Value.html) formula,
but I wanted to highlight NPV because it is far more adaptable to all kinds of situations._  
_2. You can use any spreadsheet software to easily compute NPV for any situation. [Here's the function for Google Sheets.](https://support.google.com/docs/answer/3093184?hl=en)_


NPV is useful in to value any situation with expected cashflows. The core question becomes; "What's the discount rate?" 

**A note on valuing stock:**  
Valuing stock is a topic that's _far_ to deep and subjective for me to cover. I recommend using NPV to take into account your
particular vesting terms as shown above, but share price is a whole other subject.

My only advice is if the company is privately held (shares not traded on a stock exchange), then the share value your
employer quotes you is _probably_ highly inflated and over-optimistic.

### Less tangible perks
This is compensation where you don't have as good of an idea about what the cashflows will look like, and a bit more
subjectivity comes in to play.

- Expense reimbursements
- Food & drinks
- Employee discounts

My rule of thumb for valuing them is pretty simple fortunately. I look at how much I'll actually use or how much they'll save me.
For example, if the employer offers employee discounts on things I don't buy anyways, then the value for that perk is $0 to me.

If I see myself getting \\$500 reimbursed to take some professional development course, then I'll use \\$500.  

### Intangible Perks
These are the most subjective perks, and are often valuable because of your personal situation even though your employer
doesnt pay expenses for them.

- Work / life Balance
- Professional growth opportunities
- Commute time
- Company culture

Some of them can be obvious. Is not having to drive 2 hours every day worth $2,000 less in annual salary? Hell yes! 

Other's aren't though. What kind of culture do you want to work in? Do you value professional growth more than getting
paid a lot right now? Everyone is going to be different, so only you can decide.


## Adding it all up

The total financial value of a compensation package is:

Total Value = Base Salary + $\sum$Highly Tangible Perks + $\sum$Less Tangible Perks 

You must consider the personal value of the intangible perks too, but there's no nice way of integrating that with an equation.

You should also take **cost of living** into account. The financial value above is your **revenue**, but what's really
important is your **net income**. Earning \\$10,000 less could still be a better deal if it means saving \\$15,000 per
year in housing and travel costs. 

**Net Income = Revenue - Living Expenses & Commuting Expenses**

---

## Salary sharing

Having market data available is important because it tends towards fair and competitive compensation for everyone.
[Salary sharing is good.](https://londontechpay.ca/about) It doesnt't necessarily mean sharing how much you make with
every coworker, but participating in salary surveys opens up that crucial market information.


[StackOverflow](https://insights.stackoverflow.com/survey/2017) is a great source of broad-market information, as are other
salary information aggregation sites.

Finding region-specific data that isn't aggregated can be difficult if you don't live in a giant city though.
In my city of London, Ontario, I've built a salary sharing tool named [OpenPay London](https://londontechpay.ca/) to try
and fix that.

[![https://londontechpay.ca/static/opengraph-header.png](https://londontechpay.ca/static/opengraph-header.png)]()

It is open-source on [Github](https://github.com/olestourko/open-tech-pay), and can easily be adapted to any other region.