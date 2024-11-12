import numpy as np

def make_contact_matrix(nchain, intra_all=False, inter_all=False, contact_string=None):
    """
    Calculate a matrix of inter/intra chain contact indicators
    
    Parameters:
        nchain (int, required): How many chains are in this design 
        
        contact_str (str, optional): String denoting how to define contacts, comma delimited between pairs of chains
            '!' denotes repulsive, '&' denotes attractive
    """
    alphabet   = [a for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    letter2num = {a:i for i,a in enumerate(alphabet)}
    
    contacts   = np.zeros((nchain, nchain))
    written    = np.zeros((nchain, nchain))
    
    
    # intra_all - everything on the diagonal has contact potential
    if intra_all:
        contacts[np.arange(nchain), np.arange(nchain)] = 1
    
    # inter all - everything off the diagonal has contact potential
    if inter_all:
        mask2d = np.full_like(contacts, False)
        for i in range(len(contacts)):
            for j in range(len(contacts)):
                if i != j:
                    mask2d[i, j] = True
        
        contacts[mask2d.astype(bool)] = 1


    # custom contacts/repulsions from user 
    if contact_string is not None:
        contact_list = contact_string.split(',')
        for c in contact_list:
            assert len(c) == 3
            i, j = letter2num[c[0]], letter2num[c[2]]

            symbol = c[1]

            assert symbol in ['!', '&']
            if symbol == '!':
                contacts[i, j] = -1
                contacts[j, i] = -1
            else:
                contacts[i, j] = 1
                contacts[j, i] = 1
            
    return contacts 

# Test the function
def test_make_contact_matrix():
    # Test 1: No special contact options, 6 chains
    print("Test 1: No special contact options, 6 chains")
    matrix = make_contact_matrix(nchain=6)
    print("Matrix size:", matrix.shape)
    print(matrix)

    # Test 2: Intra-chain contact, 6 chains
    print("\nTest 2: Intra-chain contact, 6 chains")
    matrix = make_contact_matrix(nchain=6, intra_all=True)
    print("Matrix size:", matrix.shape)
    print(matrix)

    # Test 3: Inter-chain contact, 6 chains
    print("\nTest 3: Inter-chain contact, 6 chains")
    matrix = make_contact_matrix(nchain=6, inter_all=True)
    print("Matrix size:", matrix.shape)
    print(matrix)

    # Test 4: Custom contacts, 6 chains
    print("\nTest 4: Custom contacts, 6 chains")
    matrix = make_contact_matrix(nchain=6, contact_string="A&B,A!C,A!E,A&F,B&C,B!D,B!F,C&D,C!E,D&E,D!F,E&F")
    print("Matrix size:", matrix.shape)
    print(matrix)

if __name__ == "__main__":
    test_make_contact_matrix()
