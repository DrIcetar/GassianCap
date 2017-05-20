class test(object):
    '''description of class'''
    objNum = 0
    def __init__(self):
        print('obj %d created' % (test.objNum + 1))

    def __del__(self):
        print('obj %d deleted' % (test.objNum - 1))

    def dispClass(self):
        print('__doc__:',test.__doc__,'\n')
        print('__name__:',test.__name__,'\n')
        print('__module__:',test.__module__,'\n')
        print('__bases__:',test.__bases__,'\n')
        print('__dict__:',test.__dict__,'\n')

t = test()
t.dispClass()
