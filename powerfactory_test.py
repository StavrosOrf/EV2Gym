import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP1\Python\3.10")

if __name__ == "__main__":
    import powerfactory as pf

    #get installation directory of powerfactory
    print(pf.__file__)

    app = pf.GetApplication()
    if app is None:
        raise Exception('getting Powerfactory application failed')
    
    #define project name and study case    
    projName = '_TSI_Nine-bus System'
    study_case = '01_Study_Case.IntCase'

    #activate project
    project = app.ActivateProject(projName)
    proj = app.GetActiveProject()

    #get the study case folder and activate project
    oFolder_studycase = app.GetProjectFolder('study')
    oCase = oFolder_studycase.GetContents(study_case)[0]
    oCase.Activate()

    #get load flow object and execute
    oLoadflow=app.GetFromStudyCase('ComLdf') #get load flow object
    oLoadflow.Execute() #execute load flow

    #get the generators and their active/reactive power and loading
    Generators = app.GetCalcRelevantObjects('*.ElmSym')
    for gen in Generators: #loop through list
        name = getattr(gen, 'loc_name') # get name of the generator
        actPower = getattr(gen,'c:p') #get active power
        reacPower = getattr(gen,'c:q') #get reactive power
        genloading = getattr(gen,'c:loading') #get loading
        #print results
        print('%s: P = %.2f MW, Q = %.2f MVAr, loading = %.0f percent' %(name,actPower,reacPower,genloading))

    print('-----------------------------------------')

    #get the lines and print their loading
    Lines=app.GetCalcRelevantObjects('*.ElmLne')
    for line in Lines: #loop through list
        name = getattr(line, 'loc_name') # get name of the line
        value = getattr(line, 'c:loading') #get value for the loading
        #print results
        print('Loading of the line: %s = %.2f percent' %(name,value))

    print('-----------------------------------------')

    #get the buses and print their voltage
    Buses=app.GetCalcRelevantObjects('*.ElmTerm')
    for bus in Buses: #loop through list
        name = getattr(bus, 'loc_name') # get name of the bus
        amp = getattr(bus, 'm:u1') #get voltage magnitude
        phase = getattr(bus, 'm:phiu') #get voltage angle
        #print results
        print('Voltage at %s = %.2f pu %.2f deg' %(name,amp,phase))