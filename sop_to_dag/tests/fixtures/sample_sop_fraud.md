# Fraud Dispute Processing SOP

## Purpose
This SOP outlines the process for handling indirect fraud disputes reported by customers.

## Procedure

### Step 1: Start Processing
Begin processing an indirect dispute.

### Step 2: Check Dispute Code
Is the dispute code 183 or 186?
- **If Yes**: Proceed to check if the borrower or co-borrower mentions fraud.
- **If No**: Proceed to check force memo for indirect disputes.

### Step 3: Check Force Memo (Indirect)
Do the FCRA notes on the ACH mention \USBFraudDSLO, \USBDS Chargebacks, or include a police report?
- **If Yes**: This is a fraud dispute (indirect). Proceed to resolve as fraud.
- **If No**: This is NOT a fraud dispute. Proceed to resolve as non-fraud.

### Step 4: Check Borrower Mention (Indirect)
Does the borrower or co-borrower mention \USBFraudDSLO or \USBDS Chargebacks or any key fraud indicators?
- **If Yes**: This is a fraud dispute (indirect). Proceed to resolve as fraud.
- **If No**: This is NOT a fraud dispute. Proceed to resolve as non-fraud.

### Step 5: Resolve as Fraud Dispute
Mark the dispute as a confirmed fraud case. Follow the Fraud Resolution Guide for next steps.
- Refer to: **Fraud Resolution Guide**
- Update the case status to "Fraud Confirmed"
- End processing.

### Step 6: Resolve as Non-Fraud Dispute
Mark the dispute as non-fraud. Follow standard dispute resolution procedures.
- Update the case status to "Non-Fraud"
- End processing.

### Step 7: Refer to External Guide
If the SOP mentions an external guide (e.g., "Refer to the Credit Bureau Reporting Disputes Operator Guide"), look up that guide.
- Use the external reference to continue processing.

## Entry Point
The very first node in the workflow MUST have the id 'start'.
