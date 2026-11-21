"""Creates sample_statement.pdf - a made-up bank statement to test the app with.

Run with:  python make_sample_pdf.py
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# (date, description, debit, credit) - all values are invented.
TRANSACTIONS = [
    ("01/09/2026", "SALARY CREDIT SEP", "", "65,000.00"),
    ("02/09/2026", "RENT PAYMENT NEFT", "18,000.00", ""),
    ("03/09/2026", "UPI-SWIGGY ORDER", "450.00", ""),
    ("05/09/2026", "AMAZON PAY SHOPPING", "2,499.00", ""),
    ("06/09/2026", "ELECTRICITY BILL MSEDCL", "1,320.00", ""),
    ("08/09/2026", "UBER TRIP", "310.00", ""),
    ("10/09/2026", "ATM WITHDRAWAL", "5,000.00", ""),
    ("12/09/2026", "UPI-ZOMATO ORDER", "620.00", ""),
    ("15/09/2026", "HOME LOAN EMI", "12,500.00", ""),
    ("16/09/2026", "NETFLIX SUBSCRIPTION", "649.00", ""),
    ("18/09/2026", "MOBILE RECHARGE", "299.00", ""),
    ("20/09/2026", "FLIPKART SHOPPING", "1,899.00", ""),
    ("22/09/2026", "IRCTC TICKET", "780.00", ""),
    ("25/09/2026", "UPI-RAHUL TRANSFER", "2,000.00", ""),
    ("28/09/2026", "INTEREST CREDIT", "", "212.00"),
    ("01/10/2026", "SALARY CREDIT OCT", "", "65,000.00"),
    ("02/10/2026", "RENT PAYMENT NEFT", "18,000.00", ""),
    ("04/10/2026", "UPI-SWIGGY ORDER", "380.00", ""),
    ("06/10/2026", "DMART GROCERY", "3,150.00", ""),
    ("07/10/2026", "ELECTRICITY BILL MSEDCL", "1,410.00", ""),
    ("09/10/2026", "OLA RIDE", "265.00", ""),
    ("11/10/2026", "MYNTRA SHOPPING", "2,750.00", ""),
    ("13/10/2026", "ATM WITHDRAWAL", "4,000.00", ""),
    ("15/10/2026", "HOME LOAN EMI", "12,500.00", ""),
    ("17/10/2026", "BROADBAND BILL", "799.00", ""),
    ("18/10/2026", "LIC PREMIUM", "5,200.00", ""),
    ("19/10/2026", "UPI-ZOMATO ORDER", "540.00", ""),
    ("21/10/2026", "PETROL PUMP", "1,500.00", ""),
    ("24/10/2026", "CAFE COFFEE DAY", "360.00", ""),
    ("27/10/2026", "UPI-PRIYA RECEIVED", "", "1,500.00"),
    ("29/10/2026", "APOLLO PHARMACY", "890.00", ""),
    ("01/11/2026", "SALARY CREDIT NOV", "", "65,000.00"),
    ("02/11/2026", "RENT PAYMENT NEFT", "18,000.00", ""),
    ("03/11/2026", "UPI-SWIGGY ORDER", "510.00", ""),
    ("05/11/2026", "AMAZON PAY SHOPPING", "4,200.00", ""),
    ("06/11/2026", "ELECTRICITY BILL MSEDCL", "1,280.00", ""),
    ("07/11/2026", "BIGBASKET ORDER", "1,150.00", ""),
    ("08/11/2026", "METRO CARD RECHARGE", "500.00", ""),
    ("09/11/2026", "SIP MUTUAL FUND", "5,000.00", ""),
    ("10/11/2026", "ATM WITHDRAWAL", "3,000.00", ""),
    ("12/11/2026", "RESTAURANT DINNER", "1,850.00", ""),
    ("14/11/2026", "MEDICAL STORE", "640.00", ""),
    ("15/11/2026", "HOME LOAN EMI", "12,500.00", ""),
]


def build_rows():
    """Add a running balance, plus one duplicate row to prove cleaning works."""
    balance = 25000.00
    rows = []
    for date, description, debit, credit in TRANSACTIONS:
        balance -= float(debit.replace(",", "") or 0)
        balance += float(credit.replace(",", "") or 0)
        rows.append([date, description, debit, credit, f"{balance:,.2f}"])
    rows.insert(6, rows[5])  # the bank printed the UBER TRIP line twice
    return rows


def main(path="sample_statement.pdf"):
    styles = getSampleStyleSheet()
    table = Table(
        [["Date", "Description", "Debit", "Credit", "Balance"]] + build_rows(),
        colWidths=[70, 200, 75, 75, 85],
        repeatRows=1,  # repeat the header on every page, like real statements
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(
        [
            Paragraph("Sample Bank - Account Statement (demo data)", styles["Title"]),
            Paragraph("Period: 01/09/2026 to 15/11/2026", styles["Normal"]),
            Spacer(1, 12),
            table,
        ]
    )
    print(f"Created {path}")


if __name__ == "__main__":
    main()
