from copy import deepcopy


class LinkGroups:
    def __init__(self, defalut_class_joint) -> None:
        self.dof = 0
        self.class_joint = defalut_class_joint

    def create_branch(self, n: int) -> list:
        return [deepcopy(self.class_joint) for __ in range(n)]

    def get_all_groups(self) -> list:
        props_class = dir(self)
        groups_methods = [
            method for method in props_class if method.startswith("get_group")
        ]
        groups = []
        for method in groups_methods:
            groups.append(getattr(self, method)())
        return groups


class LinkGroups1D(LinkGroups):
    def __init__(self, defalut_class_joints) -> None:
        super().__init__(defalut_class_joints)
        self.dof = 1

    def get_group_3n4p(self) -> tuple[list, tuple]:
        return [self.create_branch(6)], ([0,0], [0,5])

    def get_group_5n7p(self) -> tuple[list, tuple]:
        branchs = []
        br1 = self.create_branch(7)
        branchs.append(br1)
        br2 = [[br1[2], br1[3]]] + self.create_branch(2) + [[br1[4], br1[5]]]
        branchs.append(br2)
        return branchs, ([0,0], [0,6])


class LinkGroupsM1D(LinkGroups):
    def __init__(self, defalut_class_joints) -> None:
        super().__init__(defalut_class_joints)
        self.dof = -1

    def get_group_1n2p(self) -> tuple[list, tuple]:
        return [self.create_branch(4)], ([0,0], [0,3])

    def get_group_3n5p(self) -> tuple[list, tuple]:
        br1 = self.create_branch(6)
        br2 = [[br1[2], br1[3]]] + self.create_branch(2)
        return [br1, br2], ([0,0], [0,5], [1, 2])

    def get_group_5n8p(self) -> tuple[list, tuple]:
        br1 = self.create_branch(7)
        br2 = [[br1[2], br1[3]]] + self.create_branch(2) + [[br1[4], br1[5]]]
        br3 = [[br2[2], br2[3]]] + self.create_branch(2)
        return [br1, br2, br3], ([0,0], [0,6], [2, 2])


class LinkGroups0D(LinkGroups):
    def __init__(self, defalut_class_joint) -> None:
        super().__init__(defalut_class_joint)
        self.dof = 0

    def get_group_2n3p(self) -> tuple[list, tuple]:
        return [self.create_branch(5)], ([0,0], [0,2])

    def get_group_4n6p_v1(self) -> tuple[list, tuple]:
        br1 = self.create_branch(6)
        br2 = [[br1[2], br1[3]]] + self.create_branch(3)
        return [br1, br2], ([0,0], [0,5], [1,3])

    def get_group_4n6p_v2(self) -> tuple[list, tuple]:
        br1 = self.create_branch(6)
        br2 = [[br1[1], br1[2]]] + self.create_branch(2) + [[br1[3], br1[4]]]
        return [br1, br2], ([0,0], [0,5])

    def get_group_6n9p(self) -> tuple[list, tuple]:
        br1 = self.create_branch(6)
        br2 = [[br1[2], br1[3]]] + self.create_branch(4)
        br3 = [[br2[1], br2[2]]] + self.create_branch(3)
        return [br1, br2, br3], ([0,0], [0, 5], [1,4], [2, 3])


class LinkGroupsM2D(LinkGroups):
    def __init__(self, defalut_class_joint) -> None:
        super().__init__(defalut_class_joint)
        self.dof = -2

    def get_group_2n4p(self) -> tuple[list, tuple]:
        br1 = self.create_branch(5)
        br2 = [[br1[1], br1[2]]] + self.create_branch(2)
        return [br1, br2], ([0,0], [0,4], [1,2])
