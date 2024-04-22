import dartpy as dart
import xml.etree.ElementTree as ET
import numpy as np

def MakeWeldJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint):
    joint_prop = dart.dynamics.getWeldJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    return joint_prop

def MakeFreeJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint, damping):
    joint_prop = dart.dynamics.FreeJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = False
    joint_prop.mVelocityLowerLimits = np.ones(6) * -100
    joint_prop.mVelocityUpperLimits = np.ones(6) * 100
    joint_prop.mDampingCoefficients = np.ones(6) * damping
    return joint_prop

def MakeRevoluteJointProperties(name, axis, T_ParentBodyToJoint, T_ChildBodyToJoint, lower, upper, damping, friction, stiffness):
    joint_prop = dart.dynamics.RevoluteJointProperties()
    joint_prop.mName = name
    joint_prop.mAxis = axis
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = True
    joint_prop.mPositionLowerLimits = np.ones(1) * lower
    joint_prop.mPositionUpperLimits = np.ones(1) * upper

    joint_prop.mVelocityLowerLimits = np.ones(1) * -100
    joint_prop.mVelocityUpperLimits = np.ones(1) * 100

    joint_prop.mForceLowerLimits = np.ones(1) * -10000.0
    joint_prop.mForceUpperLimits = np.ones(1) * 10000.0

    joint_prop.mDampingCoefficients = np.ones(1) * damping
    joint_prop.mFrictions = np.ones(1) * friction
    joint_prop.mSpringStiffnesses = np.ones(1) * stiffness
    return joint_prop

def MakeBallJointProperties(name, T_ParentBodyToJoint, T_ChildBodyToJoint, lower, upper, damping, friction, stiffness = np.zeros(3)):
    joint_prop = dart.dynamics.BallJointProperties()
    joint_prop.mName = name
    joint_prop.mT_ParentBodyToJoint = T_ParentBodyToJoint
    joint_prop.mT_ChildBodyToJoint = T_ChildBodyToJoint
    joint_prop.mIsPositionLimitEnforced = True
    joint_prop.mPositionLowerLimits = lower
    joint_prop.mPositionUpperLimits = upper

    joint_prop.mVelocityLowerLimits = np.ones(3) * -100
    joint_prop.mVelocityUpperLimits = np.ones(3) * 100

    joint_prop.mForceLowerLimits = np.ones(3) * -10000.0
    joint_prop.mForceUpperLimits = np.ones(3) * 10000.0

    joint_prop.mDampingCoefficients = np.ones(3) * damping
    joint_prop.mFrictions = np.ones(3) * friction
    joint_prop.mSpringStiffnesses = stiffness
    return joint_prop



def MakeBodyNode(skel, parent, joint_properties, joint_type, intertia):
    if joint_type == "Free":
        [joint, body] = skel.createFreeJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "Revolute":
        [joint, body] = skel.createRevoluteJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "Ball":
        [joint, body] = skel.createBallJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))
    elif joint_type == "Weld":
        [joint, body] = skel.createWeldJointAndBodyNodePair(parent, joint_properties, dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties(joint_properties.mName)))

    body.setInertia(intertia)
    return body

def Orthonormalize(T):

    v0 = T.rotation()[:, 0]
    v1 = T.rotation()[:, 1]
    v2 = T.rotation()[:, 2]

    u0 = v0
    u1 = v1 - np.dot(u0, v1) / np.dot(u0, u0) * u0
    u2 = v2 - np.dot(u0, v2) / np.dot(u0, u0) * u0 - np.dot(u1, v2) / np.dot(u1, u1) * u1

    res = np.zeros([3,3])
    res[:, 0] = u0 / np.linalg.norm(u0)
    res[:, 1] = u1 / np.linalg.norm(u1)
    res[:, 2] = u2 / np.linalg.norm(u2)

    T.set_rotation(res)
    return T

# XML file to Skeleton
def buildFromFile(path = None, defaultDamping = 0.2):
    
    if path is not None:
        doc = ET.parse(path)
        # Error handling
        if doc is None:
            print("File not found")
            return None
        
        root = doc.getroot()
        skel = dart.dynamics.Skeleton(root.attrib['name'])

        bvh_info = {}
        joints_pd_gain = []

        for node in root:
            name = node.attrib['name']
            parent_str = node.attrib['parent']
            parent = None
            if parent_str != "None":
                parent = skel.getBodyNode(parent_str)
        
            
            T_body = dart.math.Isometry3().Identity()
            type = node.find("Body").attrib['type']
            mass = float(node.find("Body").attrib['mass'])
            
            shape = None
            if type == "Box":
                size = np.array(node.find("Body").attrib['size'].strip().split(' ')).astype(np.float32)
                shape = dart.dynamics.BoxShape(size)
            else:
                print("Not implemented")
                return None
        
            ## contact
            contact = node.find("Body").attrib['contact'] == "On"
            color = np.ones(4) * 0.2
            if 'color' in node.find("Body").attrib:
                color = np.array(node.find("Body").attrib['color'].split(' ')).astype(np.float32)


            inertia = dart.dynamics.Inertia()
            inertia.setMoment(shape.computeInertia(mass))
            inertia.setMass(mass)
            
            T_body = dart.math.Isometry3().Identity()
            T_body.set_rotation(np.array(node.find("Body").find("Transformation").attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3))
            T_body.set_translation(np.array(node.find("Body").find("Transformation").attrib['translation'].strip().split(' ')).astype(np.float32))
            T_body = Orthonormalize(T_body)
            
            joint = node.find("Joint")
            type = joint.attrib['type']
            if 'bvh' in joint.attrib:
                bvh_info[name] = joint.attrib['bvh']

            T_joint = dart.math.Isometry3().Identity()
            T_joint.set_rotation(np.array(node.find("Joint").find("Transformation").attrib['linear'].strip().split(' ')).astype(np.float32).reshape(3,3))
            T_joint.set_translation(np.array(node.find("Joint").find("Transformation").attrib['translation'].strip().split(' ')).astype(np.float32))
            T_joint = Orthonormalize(T_joint)
            
            parent_to_joint = T_joint
            if parent != None:
                parent_to_joint = parent.getTransform().inverse().multiply(T_joint)
            
            child_to_joint = T_body.inverse().multiply(T_joint)

            props = None
            if type == "Free":
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                props = MakeFreeJointProperties(name, parent_to_joint, child_to_joint, damping)
            elif type == "Ball":
                lower = np.array(joint.attrib['lower'].strip().split(' ')).astype(np.float32)
                upper = np.array(joint.attrib['upper'].strip().split(' ')).astype(np.float32)
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                friction = 0.0
                if 'friction' in joint.attrib:
                    friction = float(joint.attrib['friction'])
                stiffness = np.zeros(3)
                if 'stiffness' in joint.attrib:
                    stiffness = np.array(joint.attrib['stiffness'].strip().split(' ')).astype(np.float32)
                props = MakeBallJointProperties(name, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness)
            elif type == "Revolute":
                axis = np.array(joint.attrib['axis'].strip().split(' ')).astype(np.float32)
                lower = float(joint.attrib['lower'])
                upper = float(joint.attrib['upper'])
                damping = defaultDamping
                if 'damping' in joint.attrib:
                    damping = float(joint.attrib['damping'])
                friction = 0.0
                if 'friction' in joint.attrib:
                    friction = float(joint.attrib['friction'])
                stiffness = 0.0
                if 'stiffness' in joint.attrib:
                    stiffness = float(joint.attrib['stiffness'])
                props = MakeRevoluteJointProperties(name, axis, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness)
            elif type == "Weld":
                props = MakeWeldJointProperties(name, parent_to_joint, child_to_joint)
            else:
                print("Not implemented")
                return None
            
            pd_gain = None 
            if 'kp' in joint.attrib:
                pd_gain = [np.array(joint.attrib['kp'].split(' ')).astype(np.float32)]
                if 'kv' in joint.attrib:
                    pd_gain.append(np.array(joint.attrib['kv'].split(' ')).astype(np.float32))
                else:
                    pd_gain.append(np.sqrt(pd_gain[0] * 2))
            joints_pd_gain.append(pd_gain)
            
            bn = MakeBodyNode(skel, parent, props, type, inertia)
            shape_node = bn.createShapeNode(shape)
            shape_node.createVisualAspect().setColor(color)
            shape_node.createDynamicsAspect()
            if contact:    
                shape_node.createCollisionAspect()

        return skel , bvh_info, joints_pd_gain  