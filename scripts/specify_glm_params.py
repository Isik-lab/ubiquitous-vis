import json

glm_social = dict()
glm_social['run_groups'] = {'123':[1,2,3],'1':[1],'2': [2],'3':[3],'13':[1,3],'23':[2,3],'12':[1,2]}
glm_social['contrasts'] = {'interact':'interact-fixation',
             'no_interact': 'no_interact-fixation',
             'interact-no_interact': 'interact-no_interact',
             'interact&no_interact': 'interact&no_interact',
             'fixation':'fixation'}
glm_social['n_runs'] = 3
json_info = json.dumps(glm_social)
f = open("glm_SIpointlights.json","w")
f.write(json_info)
f.close()

glm_language = dict()
glm_language['run_groups'] = {'12':[1,2],'1':[1],'2': [2]}
glm_language['contrasts'] = {'intact':'intact-fixation',
             'degraded': 'degraded-fixation',
             'intact-degraded': 'intact -degraded',
             'degraded-intact': 'degraded-intact',
             'fixation':'fixation'}
glm_language['n_runs'] = 2
json_info = json.dumps(glm_language)
f = open("glm_language.json","w")
f.write(json_info)
f.close()
