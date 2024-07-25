import json
import os
import time
import torch
from sae_lens import HookedSAETransformer, SAE

MODEL_ID = 'meta-llama/Meta-Llama-3-8B'

with open('.credentials.json') as f:
    creds = json.load(f)
    
os.environ['HF_TOKEN'] = creds['HF_TOKEN']

text = """Amazon S3 Standard, S3 Standard-Infrequent Access, S3 Intelligent-Tiering, S3 Glacier Instant Retrieval, S3 Glacier Flexible Retrieval, and S3 Glacier Deep Archive storage classes replicate data across a minimum of three AZs to protect against the loss of one entire AZ. This remains true in Regions where fewer than three AZs are publicly available. Objects stored in these storage classes are available for access from all of the AZs in an AWS Region.
The Amazon S3 One Zone-IA storage class replicates data within a single AZ. The data stored in S3 One Zone-IA is not resilient to the physical loss of an Availability Zone resulting from disasters, such as earthquakes, fires, and floods.
Q:  How do I decide which AWS Region to store my data in?

There are several factors to consider based on your specific application. For instance, you may want to store your data in a Region that is near your customers, your data centers, or other AWS resources to reduce data access latencies. You may also want to store your data in a Region that is remote from your other operations for geographic redundancy and disaster recovery purposes. You should also consider Regions that let you address specific legal and regulatory requirements and/or reduce your storage costs—you can choose a lower priced Region to save money. For S3 pricing information, visit the Amazon S3 pricing page.

Q:  In which parts of the world is Amazon S3 available?

Amazon S3 is available in AWS Regions worldwide, and you can use Amazon S3 regardless of your location. You just have to decide which AWS Region(s) you want to store your Amazon S3 data. See the AWS regional services list for a list of AWS Regions in which S3 is available today.

Billing
Q:  How much does Amazon S3 cost?

With Amazon S3, you pay only for what you use. There is no minimum charge. You can estimate your monthly bill using the AWS Pricing Calculator.

AWS charges less where our costs are less. Some prices vary across Amazon S3 Regions. Billing prices are based on the location of your S3 bucket. There is no Data Transfer charge for data transferred within an Amazon S3 Region via a COPY request. Data transferred via a COPY request between AWS Regions is charged at rates specified on the Amazon S3 pricing page. There is no Data Transfer charge for data transferred between Amazon EC2 (or any AWS service) and Amazon S3 within the same Region, for example, data transferred within the US East (Northern Virginia) Region. However, data transferred between Amazon EC2 (or any AWS service) and Amazon S3 across all other Regions is charged at rates specified on the Amazon S3 pricing page, for example, data transferred between Amazon EC2 US East (Northern Virginia) and Amazon S3 US West (Northern California). Data transfer costs are billed to the source bucket owner.

For S3 on Outposts pricing, visit the Outposts pricing page.

Q:  How will I be charged and billed for my use of Amazon S3?

There are no set up charges or commitments to begin using Amazon S3. At the end of the month, you will automatically be charged for that month’s usage. You can view your charges for the current billing period at any time by logging into your Amazon Web Services account, and selecting the 'Billing Dashboard' associated with your console profile.

With the AWS Free Usage Tier*, you can get started with Amazon S3 for free in all Regions except the AWS GovCloud Regions. Upon sign up, new AWS customers receive 5 GB of Amazon S3 Standard storage, 20,000 Get Requests, 2,000 Put Requests, and 100 GB of data transfer out (to internet, other AWS Regions, or Amazon CloudFront) each month for one year. Unused monthly usage will not roll over to the next month.

Amazon S3 charges you for the following types of usage. Note that the calculations below assume there is no AWS Free Tier in place."""

transformer = HookedSAETransformer.from_pretrained(MODEL_ID, device='cuda')


bs = [1, 4]
times = []
for i in bs:
    print(i)
    sentences = [text] * i
    output = transformer.tokenizer(sentences, padding='max_length', truncation=True, max_length=64,  return_tensors='pt')
    input_ids = output['input_ids'].to('cuda')


    start = time.time()

    for i in range(10):
        _, all_hidden_states = transformer.run_with_cache(
            input_ids, 
            prepend_bos=True, 
            stop_at_layer=8 + 1
        )

        
    end = time.time()
    print(all_hidden_states.shape)
    times.append(end - start)

print(bs)
print(times)


