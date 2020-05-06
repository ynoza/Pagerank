import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    ret={}
    if corpus.get(page):
        length=len(corpus[page])
        for k in corpus[page]:
            ret.update({k:damping_factor*(1/length)})
        length=len(corpus.keys())
        sum=0
        for k in corpus.keys():
            if ret.get(k):
                ret[k]+=(1/length)*(1-damping_factor)
            else:
                ret.update({k:(1/length)*(1-damping_factor)})
            sum+=ret[k]
    else:
        length=len(corpus.keys())
        sum=0
        for k in corpus.keys():
            ret.update({k:1/length})
            sum+=ret[k]


    assert(round(sum) == 1), ret

    return ret


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #create empty dictionary with all values 0
    ret=dict()
    for k in corpus.keys():
        ret[k]=0

    #find a random initial page
    length=len(corpus.keys())
    f_rand=random.randint(0,length-1)
    page=""
    count=0
    for k in corpus.keys():
        if count == f_rand:
            page=k
            break
        count+=1

    #go through n samples
    for iter in range(0,n):
        trans=transition_model(corpus,page,damping_factor)
        rand_num=random.randint(0,1000000)
        prev=0
        #using random try to pick a page based on the probablities in trans
        for k in sorted(trans.keys()):
            trans[k]+=prev
            prev=trans[k]
            if rand_num <= trans[k]*1000000:
                ret[k]+=1
                page=k
                break

    #change values in ret from count to proportion
    sum=0
    for k in ret.keys():
        ret[k]/=n
        sum+=ret[k]

    assert(round(sum) == 1)

    return ret

def iterate_pagerank_wrapper(corpus, damping_factor, ret, page, length):

    #find all pages that link to p
    lst=[]
    for k in corpus.keys():
        if page in corpus[k]:
            lst.append(k)

    val=0

    for i in lst:
        val += ret[i]/len(corpus[i])

    ret[page]= (1-damping_factor)/length + damping_factor*val

    return ret


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ret=dict()
    length=len(corpus.keys())
    for k in corpus.keys():
        ret[k]=1/length

    lst=corpus.keys()
    while(True):
        done=False
        for k in lst:
            prev=ret[k]
            ret= iterate_pagerank_wrapper(corpus, damping_factor, ret, k, length)
            if ret[k] - prev  > 0.001:
                done = True
        if done:
            break

    return ret


if __name__ == "__main__":
    main()
