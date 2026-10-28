# COMPROBACIONES DE NEGACIÓN, LISTAS, CONTEXTO Y CONTRADICCIONES.
pruebas = [
 ('ACL normal.', 'ACL','ausente'),
 ('Anterior cruciate ligament (ACL): No tear.','ACL','ausente'),
 ('Normal ACL, PCL, MCL and LCL.','MCL','ausente'),
 ('The medial meniscus and medial collateral ligament are unremarkable.','Medial Meniscus','ausente'),
 ('Intercondylar compartment (anterior and posterior cruciate ligaments):\nNormal.','ACL','ausente'),
 ('MCL; Low grade partial tear at femoral side.','MCL','presente'),
 ('ACL; High grade almost complete tear with no anterior tibial translation.','ACL','presente'),
 ('Osseous contusion without fracture.','Contusion','presente'),
 ('Osseous contusion without fracture.','Fracture','ausente'),
 ('Possible ACL tear.','ACL','incierto'),
 ('No significant joint effusion.','Effusion','incierto'),
 ('ACL normal. Tear of the ACL.','ACL','incierto'),
 ('No medial compartment OA.','Medial OA','ausente'),
 ('ACL not normal.','ACL','incierto'),
 ('No fracture.','Synovitis','no_mencionado'),
 ('Tricompartmental osteoarthritis.','Lateral OA','presente'),
]
for texto,c,esperado in pruebas:
    resultado=next(x['estado'] for x in extraer_reporte_v2(texto) if x['condicion']==c)
    assert resultado==esperado,(texto,c,resultado,esperado)
print('Comprobaciones técnicas correctas:',len(pruebas))
