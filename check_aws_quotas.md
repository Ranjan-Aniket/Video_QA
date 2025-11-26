# Check AWS Service Quotas

## Quick Check (No CLI Needed)

### Option 1: AWS Console (Easy)

1. **Go to**: https://console.aws.amazon.com/servicequotas/

2. **Search for**: "EC2"

3. **Check these quotas**:
   ```
   Running On-Demand G and VT instances
   → Shows how many g4dn/g5 vCPUs you can use
   → New accounts: Usually 0-4 (may need increase)

   Running On-Demand Standard instances
   → Shows how many t2/t3/m5/c5 vCPUs you can use
   → New accounts: Usually 32-64 vCPUs ✅
   ```

4. **For g4dn.xlarge**: You need **4 vCPUs** in "G" quota
   - If quota shows "4" or more: ✅ You can launch immediately!
   - If quota shows "0": ❌ Need to request increase

---

## Option 2: Try Launching Instance (Fastest)

1. **Go to**: https://console.aws.amazon.com/ec2/

2. **Click**: "Launch Instance"

3. **Select**:
   - AMI: "Deep Learning AMI GPU PyTorch"
   - Instance type: **g4dn.xlarge**

4. **Click**: "Launch"

**What happens:**
- ✅ **Success**: You have quota! Continue with setup
- ❌ **Error "vCPU limit exceeded"**: Need quota increase

---

## If You Need Quota Increase

### For GPU Instances (g4dn.xlarge)

1. **Go to**: https://console.aws.amazon.com/servicequotas/

2. **Search**: "Running On-Demand G and VT instances"

3. **Click**: "Request quota increase"

4. **Enter**:
   ```
   Current quota: 0
   Requested quota: 8 (allows 2x g4dn.xlarge instances)
   Reason: ML/AI video processing workload
   ```

5. **Submit**: Usually approved in 24-48 hours

---

## Alternative: Check if You Already Have Quota

Many AWS accounts actually DO have small GPU quotas. Try this:

### Quick Test Script

```bash
# Install AWS CLI
pip install awscli

# Configure (need Access Key)
aws configure

# Check quotas
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA

# Output shows your current G instance vCPU limit
```

---

## 🎯 Most Likely Scenario

**Your AWS account probably CAN launch g4dn.xlarge!**

New accounts often get:
- ✅ 4-8 vCPUs for G instances (enough for 1-2 g4dn.xlarge)
- ✅ 32+ vCPUs for standard instances
- ✅ Immediate access to t2/t3/m5 instances

**Just try launching it!** The worst that happens is an error message asking you to request quota.

---

## If GPU Quota is 0

### Option A: Request Increase (24-48 hours)
Follow steps above

### Option B: Use Different Region
Some regions have higher default quotas:
- Try: **us-east-1** (Virginia)
- Try: **us-west-2** (Oregon)

### Option C: Use AWS Credits/Activate Account
New accounts with credits often get higher quotas automatically

---

## For GCP (Similar Process)

1. **Go to**: https://console.cloud.google.com/iam-admin/quotas

2. **Filter**: "GPUs (all regions)"

3. **Check quota**: Usually 0-1 for new accounts

4. **Request increase**: Similar process, 24-48 hours

---

## 💡 Pro Tip

**Most new accounts CAN launch at least 1 g4dn.xlarge!**

Just go to EC2 console and try. If it works, you're good to go! If not, request takes 1-2 days.
