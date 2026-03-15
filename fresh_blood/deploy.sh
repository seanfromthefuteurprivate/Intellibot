#!/bin/bash
#
# Fresh Blood — Deploy to EC2
#
# Usage: ./deploy.sh
#

set -e

INSTANCE_ID="i-03f3a7c46ec809a43"
REGION="us-east-1"
REMOTE_PATH="/home/ubuntu/wsb-snake/fresh_blood"

echo "=============================================="
echo "FRESH BLOOD — DEPLOYMENT"
echo "=============================================="

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "1. Pushing to git..."
git add -A
git commit -m "Deploy Fresh Blood strategy" || true
git push origin fresh-blood

echo ""
echo "2. Deploying to EC2..."
COMMAND_ID=$(aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name "AWS-RunShellScript" \
    --parameters "commands=[
        'cd /home/ubuntu/wsb-snake',
        'git fetch origin',
        'git checkout fresh-blood',
        'git pull origin fresh-blood',
        'pip3 install requests --quiet',
        'sudo cp fresh_blood/fresh_blood.service /etc/systemd/system/',
        'sudo cp fresh_blood/fresh_blood.timer /etc/systemd/system/',
        'sudo systemctl daemon-reload',
        'sudo systemctl enable fresh_blood.timer',
        'sudo systemctl start fresh_blood.timer',
        'echo Done!'
    ]" \
    --region "$REGION" \
    --output text \
    --query "Command.CommandId")

echo "Command ID: $COMMAND_ID"
echo ""
echo "3. Waiting for deployment..."
sleep 5

# Get result
aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query "[Status, StandardOutputContent]" \
    --output text

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "To test manually:"
echo "  aws ssm send-command \\"
echo "    --instance-ids $INSTANCE_ID \\"
echo "    --document-name AWS-RunShellScript \\"
echo "    --parameters 'commands=[\"cd $REMOTE_PATH && python3 run.py scan\"]' \\"
echo "    --region $REGION"
echo ""
echo "To check timer status:"
echo "  aws ssm send-command \\"
echo "    --instance-ids $INSTANCE_ID \\"
echo "    --document-name AWS-RunShellScript \\"
echo "    --parameters 'commands=[\"systemctl status fresh_blood.timer\"]' \\"
echo "    --region $REGION"
